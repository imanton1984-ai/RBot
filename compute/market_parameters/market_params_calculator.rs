//market situation based on BTC indicators - Flat, downtrend, uppertrend. market params must contain 2 values global trend - it is longterm trend and market situation and short term - last 1-3 days.
//main target is to not trade agains trend and market and adjust leverage if it is high volatility and oppoite- and have additional score for this parameter is final scorer

// compute/scoring/market_params_calculator.rs

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{PgPool, Row};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use common::Symbol;

/// High-level market situation for BTC regime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketSituation {
    Uptrend,
    Downtrend,
    Flat,
    /// Global up, short down (pullback)
    PullbackInUptrend,
    /// Global down, short up (relief rally)
    ReliefInDowntrend,
}

impl MarketSituation {
    pub fn as_str(&self) -> &'static str {
        match self {
            MarketSituation::Uptrend => "uptrend",
            MarketSituation::Downtrend => "downtrend",
            MarketSituation::Flat => "flat",
            MarketSituation::PullbackInUptrend => "pullback_in_uptrend",
            MarketSituation::ReliefInDowntrend => "relief_in_downtrend",
        }
    }
}

/// BTC market regime parameters (used by final scorer and risk/leverage)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketParams {
    pub asof: DateTime<Utc>,

    pub btc_symbol: String,
    pub btc_symbol_id: i64,

    // timeframes used for regime
    pub global_tf_minutes: i16, // usually 1440
    pub short_tf_minutes: i16,  // usually 240

    // -1 / 0 / +1
    pub global_dir: i8,
    pub short_dir: i8,

    pub situation: MarketSituation,

    // "strength" and "risk"
    pub adx_global: Option<f32>,
    pub atr_global: Option<f32>,
    pub close_global: Option<f64>,
    pub atr_pct_global: Option<f32>, // (atr/close)*100

    pub trend_strength_score: f64, // 0..1 (from ADX)
    pub volatility_score: f64,     // 0..1 (from ATR%)

    /// Direction-neutral “quality” (helps weight tuning; not the same as alignment)
    pub market_quality_score: f64, // 0..1

    pub details_json: Value,
}

impl MarketParams {
    /// Neutral fallback when BTC data is not available yet.
    /// Returns conservative parameters that don't bias any direction.
    pub fn neutral(asof: DateTime<Utc>) -> Self {
        Self {
            asof,
            btc_symbol: "BTCUSDT".to_string(),
            btc_symbol_id: 0,
            global_tf_minutes: 1440,
            short_tf_minutes: 240,
            global_dir: 0,
            short_dir: 0,
            situation: MarketSituation::Flat,
            adx_global: None,
            atr_global: None,
            close_global: None,
            atr_pct_global: None,
            trend_strength_score: 0.0,
            volatility_score: 0.0,
            market_quality_score: 0.0,
            details_json: json!({"status": "neutral_fallback", "reason": "BTC data not available"}),
        }
    }

    /// 0..1 : how much the requested side matches BTC market regime
    /// side: +1 long, -1 short
    pub fn alignment_score(&self, side: i8) -> f64 {
        let side = side.signum();
        if side == 0 {
            return 0.0;
        }

        let g = self.global_dir.signum();
        let s = self.short_dir.signum();

        // base alignment from both horizons
        let mut score: f64 = 0.5;

        if g == side {
            score += 0.35;
        } else if g != 0 && g != side {
            score -= 0.35;
        }

        if s == side {
            score += 0.15;
        } else if s != 0 && s != side {
            score -= 0.15;
        }

        // clamp
        score.clamp(0.0, 1.0)
    }

    /// 0..1 : market component for final scorer.
    /// Penalize high volatility (risk), but not kill good trends completely.
    pub fn score_for_side(&self, side: i8) -> f64 {
        let align = self.alignment_score(side);
        let risk_ok = 1.0 - self.volatility_score; // 1 = low risk, 0 = very high vol

        // If not aligned, volatility hurts more
        let risk_factor = if align >= 0.7 {
            0.65 + 0.35 * risk_ok
        } else {
            0.35 + 0.65 * risk_ok
        };

        (align * risk_factor).clamp(0.0, 1.0)
    }

    /// 0..1 factor to scale leverage (1.0 = keep base leverage)
    pub fn leverage_factor(&self, side: i8) -> f64 {
        let align = self.alignment_score(side);
        let vol = self.volatility_score;

        // Strong cut when high vol and low alignment
        let factor = 1.0 - (vol * (1.0 - align) * 0.85);

        factor.clamp(0.15, 1.0)
    }
}

struct CacheEntry {
    at: Instant,
    params: MarketParams,
}

pub struct MarketParamsCalculator {
    btc_symbol: Symbol,
    global_tf_minutes: i16,
    short_tf_minutes: i16,

    // caching
    ttl: Duration,
    cache: RwLock<Option<CacheEntry>>,
    btc_symbol_id_cache: RwLock<Option<i64>>,
}

impl MarketParamsCalculator {
    pub fn new(btc_symbol: Symbol) -> Self {
        Self {
            btc_symbol,
            global_tf_minutes: 1440, // 1d
            short_tf_minutes: 240,   // 4h
            ttl: Duration::from_secs(60),
            cache: RwLock::new(None),
            btc_symbol_id_cache: RwLock::new(None),
        }
    }

    pub fn with_timeframes(mut self, global_tf_minutes: i16, short_tf_minutes: i16) -> Self {
        self.global_tf_minutes = global_tf_minutes;
        self.short_tf_minutes = short_tf_minutes;
        self
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    pub async fn get_or_compute(&self, pool: &PgPool, asof: DateTime<Utc>) -> Result<MarketParams> {
        // fast path: cache hit
        {
            let guard = self.cache.read().await;
            if let Some(entry) = guard.as_ref() {
                if entry.at.elapsed() <= self.ttl {
                    return Ok(entry.params.clone());
                }
            }
        }

        let params = match self.compute(pool, asof).await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    "MarketParamsCalculator: failed to compute BTC regime ({}), using neutral fallback",
                    e
                );
                MarketParams::neutral(asof)
            }
        };

        {
            let mut guard = self.cache.write().await;
            *guard = Some(CacheEntry {
                at: Instant::now(),
                params: params.clone(),
            });
        }

        Ok(params)
    }

    pub async fn compute(&self, pool: &PgPool, asof: DateTime<Utc>) -> Result<MarketParams> {
        let btc_symbol_id = self.resolve_btc_symbol_id(pool).await?;

        // Global (1d by default): need EMA50/EMA200 slope + last close + ADX/ATR
        let global_ind = fetch_recent_indicators(pool, btc_symbol_id, self.global_tf_minutes, 12).await?;
        let global_close = fetch_last_close(pool, btc_symbol_id, self.global_tf_minutes).await?;

        let (global_dir, adx_global, atr_global, atr_pct_global, trend_strength_score, volatility_score) =
            derive_global_components(&global_ind, global_close);

        // Short (4h by default): use trend_short if present, fallback to EMA200 slope check
        let short_ind = fetch_recent_indicators(pool, btc_symbol_id, self.short_tf_minutes, 24).await?;
        let short_dir = derive_short_dir(&short_ind);

        let situation = derive_situation(global_dir, short_dir);

        // direction-neutral quality: prefer higher ADX, prefer lower vol (risk)
        let market_quality_score = (0.60 * trend_strength_score + 0.40 * (1.0 - volatility_score)).clamp(0.0, 1.0);

        let details_json = json!({
            "btc_symbol": self.btc_symbol.0,
            "btc_symbol_id": btc_symbol_id,
            "global_tf_minutes": self.global_tf_minutes,
            "short_tf_minutes": self.short_tf_minutes,
            "global_dir": global_dir,
            "short_dir": short_dir,
            "situation": situation.as_str(),
            "adx_global": adx_global,
            "atr_global": atr_global,
            "close_global": global_close,
            "atr_pct_global": atr_pct_global,
            "trend_strength_score": trend_strength_score,
            "volatility_score": volatility_score,
            "market_quality_score": market_quality_score,
        });

        Ok(MarketParams {
            asof,
            btc_symbol: self.btc_symbol.0.clone(),
            btc_symbol_id,
            global_tf_minutes: self.global_tf_minutes,
            short_tf_minutes: self.short_tf_minutes,
            global_dir,
            short_dir,
            situation,
            adx_global,
            atr_global,
            close_global: global_close,
            atr_pct_global,
            trend_strength_score,
            volatility_score,
            market_quality_score,
            details_json,
        })
    }

    async fn resolve_btc_symbol_id(&self, pool: &PgPool) -> Result<i64> {
        // cached
        {
            let g = self.btc_symbol_id_cache.read().await;
            if let Some(id) = *g {
                return Ok(id);
            }
        }

        let row = sqlx::query("SELECT symbol_id FROM market.pairs WHERE symbol = $1 LIMIT 1")
            .bind(&self.btc_symbol.0)
            .fetch_one(pool)
            .await
            .with_context(|| format!("resolve BTC symbol_id for {}", self.btc_symbol.0))?;

        let id: i64 = row.try_get("symbol_id")?;

        {
            let mut g = self.btc_symbol_id_cache.write().await;
            *g = Some(id);
        }

        Ok(id)
    }
}

fn candles_table(tf_minutes: i16) -> &'static str {
    match tf_minutes {
        1 => "market.candles_1m",
        5 => "market.candles_5m",
        15 => "market.candles_15m",
        60 => "market.candles_1h",
        240 => "market.candles_4h",
        1440 => "market.candles_1d",
        _ => "market.candles_1h", // safe fallback
    }
}

async fn fetch_last_close(pool: &PgPool, symbol_id: i64, tf_minutes: i16) -> Result<Option<f64>> {
    let table = candles_table(tf_minutes);
    let sql = format!(
        "SELECT close FROM {} WHERE symbol_id = $1 ORDER BY time DESC LIMIT 1",
        table
    );

    let close: Option<f64> = sqlx::query_scalar(&sql)
        .bind(symbol_id)
        .fetch_optional(pool)
        .await
        .with_context(|| format!("fetch_last_close: {} sym_id={}", table, symbol_id))?;

    Ok(close)
}

async fn fetch_recent_indicators(
    pool: &PgPool,
    symbol_id: i64,
    tf_minutes: i16,
    limit: i64,
) -> Result<Vec<(DateTime<Utc>, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<i16>, Option<i16>)>> {
    // time, ema_50, ema_200, adx, atr, trend, trend_short
    let rows = sqlx::query(
        r#"
        SELECT time, ema_50, ema_200, adx, atr, trend, trend_short
        FROM market.indicators_wide
        WHERE symbol_id = $1 AND tf_minutes = $2
        ORDER BY time DESC
        LIMIT $3
        "#,
    )
    .bind(symbol_id)
    .bind(tf_minutes)
    .bind(limit)
    .fetch_all(pool)
    .await
    .with_context(|| format!("fetch_recent_indicators sym_id={} tf={}", symbol_id, tf_minutes))?;

    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let time: DateTime<Utc> = r.try_get("time")?;
        let ema_50: Option<f32> = r.try_get("ema_50")?;
        let ema_200: Option<f32> = r.try_get("ema_200")?;
        let adx: Option<f32> = r.try_get("adx")?;
        let atr: Option<f32> = r.try_get("atr")?;
        let trend: Option<i16> = r.try_get("trend")?;
        let trend_short: Option<i16> = r.try_get("trend_short")?;
        out.push((time, ema_50, ema_200, adx, atr, trend, trend_short));
    }

    Ok(out)
}

fn derive_global_components(
    global_ind: &[(DateTime<Utc>, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<i16>, Option<i16>)],
    global_close: Option<f64>,
) -> (i8, Option<f32>, Option<f32>, Option<f32>, f64, f64) {
    // take last values
    let mut ema200_last: Option<f32> = None;
    let mut ema200_prev: Option<f32> = None;
    let mut ema50_last: Option<f32> = None;
    let mut adx_last: Option<f32> = None;
    let mut atr_last: Option<f32> = None;

    if let Some((_, ema50, ema200, adx, atr, _, _)) = global_ind.first() {
        ema50_last = *ema50;
        ema200_last = *ema200;
        adx_last = *adx;
        atr_last = *atr;
    }
    if global_ind.len() >= 2 {
        ema200_prev = global_ind[1].2;
    }

    // slope
    let slope = match (ema200_last, ema200_prev) {
        (Some(a), Some(b)) if b.abs() > 1e-9 => (a - b) / b,
        _ => 0.0,
    };

    let close = global_close.unwrap_or(0.0);

    // trend dir logic
    let mut dir = 0i8;
    if let (Some(ema200), Some(ema50)) = (ema200_last, ema50_last) {
        let above = close > ema200 as f64 && (ema50 as f64) > (ema200 as f64);
        let below = close < ema200 as f64 && (ema50 as f64) < (ema200 as f64);

        if above && slope > 0.0 {
            dir = 1;
        } else if below && slope < 0.0 {
            dir = -1;
        } else {
            dir = 0;
        }
    }

    // trend strength from ADX
    let trend_strength_score = if let Some(adx) = adx_last {
        // 15 => 0, 40 => 1
        ((adx as f64 - 15.0) / 25.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // ATR% -> volatility score
    let atr_pct = match (atr_last, global_close) {
        (Some(atr), Some(c)) if c > 0.0 => Some((atr as f64 / c) * 100.0),
        _ => None,
    };

    let volatility_score = if let Some(pct) = atr_pct {
        // 0.6% => low, 4.0% => very high
        ((pct - 0.6) / 3.4).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (dir, adx_last, atr_last, atr_pct.map(|x| x as f32), trend_strength_score, volatility_score)
}

fn derive_short_dir(
    short_ind: &[(DateTime<Utc>, Option<f32>, Option<f32>, Option<f32>, Option<f32>, Option<i16>, Option<i16>)],
) -> i8 {
    // Prefer trend_short if present
    if let Some((_, _, _, _, _, _, trend_short)) = short_ind.first() {
        if let Some(ts) = trend_short {
            return (*ts).signum() as i8;
        }
    }

    // Fallback: EMA200 slope
    let ema200_last = short_ind.first().and_then(|x| x.2);
    let ema200_prev = if short_ind.len() >= 2 { short_ind[1].2 } else { None };

    match (ema200_last, ema200_prev) {
        (Some(a), Some(b)) if a > b => 1,
        (Some(a), Some(b)) if a < b => -1,
        _ => 0,
    }
}

fn derive_situation(global_dir: i8, short_dir: i8) -> MarketSituation {
    match (global_dir.signum(), short_dir.signum()) {
        (1, 1) => MarketSituation::Uptrend,
        (-1, -1) => MarketSituation::Downtrend,
        (1, -1) => MarketSituation::PullbackInUptrend,
        (-1, 1) => MarketSituation::ReliefInDowntrend,
        _ => MarketSituation::Flat,
    }
}
