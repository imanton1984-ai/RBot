// compute/src/raw_signal_processor.rs

use std::collections::HashMap;

use common::{Symbol, Timeframe};
use serde_json::{json, Map, Value as JsonValue};

use crate::compute_backend::FeatureWindow;
use crate::raw_signal_types::RawSignal as ComputeRawSignal;
use crate::types::FeatureValue;

use raw_signals::thresholds::{RawSignal as RawSignalsRawSignal, SignalConfig};

pub struct RawSignalProcessor {
    pub config: SignalConfig,
}

impl RawSignalProcessor {
    pub fn new(config: SignalConfig) -> Self {
        Self { config }
    }

    pub fn process_feature_window(&self, feature_window: &FeatureWindow) -> Vec<ComputeRawSignal> {
        let candle_window = match &feature_window.candle_window {
            Some(w) => w,
            None => return vec![],
        };

        let n = candle_window.timestamps.len();
        if n == 0 {
            return vec![];
        }

        // Assuming a single symbol/timeframe per feature window for now.
        let Some(first_feature) = feature_window.features.get(0) else {
            return vec![]; // No features, no signals
        };
        let symbol = first_feature.symbol.clone();
        let timeframe = first_feature.timeframe;

        let timestamps = &candle_window.timestamps;
        // Create vectors for symbol and timeframe for each candle
        let symbols = vec![symbol.clone(); n];
        let timeframes = vec![timeframe; n];

        // Индекс для вытягивания features по (symbol, timeframe, timestamp)
        let mut feat_idx: HashMap<(common::Symbol, common::Timeframe, i64), usize> =
            HashMap::with_capacity(feature_window.features.len());
        for (i, r) in feature_window.features.iter().enumerate() {
            feat_idx.insert((r.symbol.clone(), r.timeframe, r.timestamp), i);
        }

        // Вытаскиваем значения одним проходом, длины всегда == n (иначе raw_signals режет массивы).
        let mut adx = vec![f64::NAN; n];
        let mut atr = vec![f64::NAN; n];
        let mut bb_upper = vec![f64::NAN; n];
        let mut bb_mid = vec![f64::NAN; n];
        let mut bb_lower = vec![f64::NAN; n];
        let mut cci = vec![f64::NAN; n];
        let mut ema20 = vec![f64::NAN; n];
        let mut ema50 = vec![f64::NAN; n];
        let mut ema200 = vec![f64::NAN; n];
        let mut macd = vec![f64::NAN; n];
        let mut macd_signal = vec![f64::NAN; n];
        let mut macd_hist = vec![f64::NAN; n];
        let mut obv = vec![f64::NAN; n];
        let mut rsi = vec![f64::NAN; n];
        let mut sma = vec![f64::NAN; n];
        let mut stoch_k = vec![f64::NAN; n];
        let mut stoch_d = vec![f64::NAN; n];
        let mut vwap = vec![f64::NAN; n];
        let mut williams = vec![f64::NAN; n];

        for i in 0..n {
            let key = (symbols[i].clone(), timeframes[i], timestamps[i]);
            let Some(&j) = feat_idx.get(&key) else { continue; };
            let feats = &feature_window.features[j].features;

            adx[i] = get_f64(feats, "adx");
            atr[i] = get_f64(feats, "atr");
            bb_upper[i] = get_f64(feats, "bb_upper");
            bb_mid[i] = get_f64(feats, "bb_mid");
            bb_lower[i] = get_f64(feats, "bb_lower");
            cci[i] = get_f64(feats, "cci");
            ema20[i] = get_f64(feats, "ema20");
            ema50[i] = get_f64(feats, "ema50");
            ema200[i] = get_f64(feats, "ema200");
            macd[i] = get_f64(feats, "macd");
            macd_signal[i] = get_f64(feats, "macd_signal");
            macd_hist[i] = get_f64(feats, "macd_hist");
            obv[i] = get_f64(feats, "obv");
            rsi[i] = get_f64(feats, "rsi");
            sma[i] = get_f64(feats, "sma");
            stoch_k[i] = get_f64(feats, "stoch_k");
            stoch_d[i] = get_f64(feats, "stoch_d");
            vwap[i] = get_f64(feats, "vwap");
            williams[i] = get_f64(feats, "williams");
        }

        let mut raw_signals: Vec<ComputeRawSignal> = Vec::new();

        // adx
        raw_signals.extend(
            raw_signals::adx_raw::calculate_adx_raw_signals(&adx, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        // atr
        raw_signals.extend(
            raw_signals::atr_raw::calculate_atr_raw_signals(&atr, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        // bb
        raw_signals.extend(
            raw_signals::bb_raw::calculate_bb_raw_signals(
                &candle_window.close,
                &bb_upper,
                &bb_mid,
                &bb_lower,
                timestamps,
                &symbols,
                &timeframes,
                &self.config,
            )
            .into_iter()
            .map(|s| self.convert_raw_signal(s)),
        );

        // cci
        raw_signals.extend(
            raw_signals::cci_raw::calculate_cci_raw_signals(&cci, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        // ema20/50/200
        raw_signals.extend(
            raw_signals::ema_raw::calculate_ema_raw_signals(&candle_window.close, &ema20, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal_with_sub_id(s, 20)),
        );
        raw_signals.extend(
            raw_signals::ema_raw::calculate_ema_raw_signals(&candle_window.close, &ema50, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal_with_sub_id(s, 50)),
        );
        raw_signals.extend(
            raw_signals::ema_raw::calculate_ema_raw_signals(&candle_window.close, &ema200, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal_with_sub_id(s, 200)),
        );

        // macd
        raw_signals.extend(
            raw_signals::macd_raw::calculate_macd_raw_signals(&macd_hist, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        // obv
        raw_signals.extend(
            raw_signals::obv_raw::calculate_obv_raw_signals(&obv, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        // rsi
        raw_signals.extend(
            raw_signals::rsi_raw::calculate_rsi_raw_signals(&rsi, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        // sma
        raw_signals.extend(
            raw_signals::sma_raw::calculate_sma_raw_signals(&candle_window.close, &sma, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        // stoch
        raw_signals.extend(
            raw_signals::stoch_raw::calculate_stoch_raw_signals(&stoch_k, &stoch_d, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        // vwap
        raw_signals.extend(
            raw_signals::vwap_raw::calculate_vwap_raw_signals(&candle_window.close, &vwap, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        // williams
        raw_signals.extend(
            raw_signals::williams_raw::calculate_williams_raw_signals(&williams, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        // Приклеиваем контекст (features_json + scores_json) по свече.
        attach_context(
            &mut raw_signals,
            candle_window,
            &symbols,
            &timeframes,
            &adx, &atr, &bb_upper, &bb_mid, &bb_lower, &cci,
            &ema20, &ema50, &ema200,
            &macd, &macd_signal, &macd_hist,
            &obv, &rsi, &sma, &stoch_k, &stoch_d, &vwap, &williams,
        );

        raw_signals
    }

    fn convert_raw_signal(&self, raw_signal: RawSignalsRawSignal) -> ComputeRawSignal {
        ComputeRawSignal {
            symbol: raw_signal.symbol,
            timeframe: raw_signal.timeframe,
            timestamp: raw_signal.timestamp,
            indicator_id: raw_signal.indicator_id,
            signal_kind: raw_signal.signal_kind,
            signal_sub_id: 0, // default (если захочешь - можно маппить по details)
            side: raw_signal.side,
            score: raw_signal.score,
            value: raw_signal.value,
            details: raw_signal.details,

            candle_is_final: true,
            calc_source: 1,
            event_time_ms: None,

            features_json: None,
            scores_json: None,
            predictions_json: None,
        }
    }

    fn convert_raw_signal_with_sub_id(&self, raw_signal: RawSignalsRawSignal, sub_id: i16) -> ComputeRawSignal {
        let mut s = self.convert_raw_signal(raw_signal);
        s.signal_sub_id = sub_id;
        s
    }
}

fn get_f64(feats: &HashMap<String, FeatureValue>, key: &str) -> f64 {
    match feats.get(key) {
        Some(FeatureValue::Float(v)) => *v,
        _ => f64::NAN,
    }
}

#[allow(clippy::too_many_arguments)]
fn attach_context(
    signals: &mut [ComputeRawSignal],
    cw: &crate::CandleWindow,
    symbols: &[Symbol],
    timeframes: &[Timeframe],
    adx: &[f64], atr: &[f64], bb_upper: &[f64], bb_mid: &[f64], bb_lower: &[f64], cci: &[f64],
    ema20: &[f64], ema50: &[f64], ema200: &[f64],
    macd: &[f64], macd_signal: &[f64], macd_hist: &[f64],
    obv: &[f64], rsi: &[f64], sma: &[f64], stoch_k: &[f64], stoch_d: &[f64], vwap: &[f64], williams: &[f64],
) {
    // индекс свечи по ключу (symbol, tf, ts)
    let mut candle_idx: HashMap<(common::Symbol, common::Timeframe, i64), usize> = HashMap::with_capacity(cw.timestamps.len());
    for i in 0..cw.timestamps.len() {
        candle_idx.insert((symbols[i].clone(), timeframes[i], cw.timestamps[i]), i);
    }

    // группируем сигналы по свече
    let mut groups: HashMap<(common::Symbol, common::Timeframe, i64), Vec<usize>> = HashMap::new();
    for (i, s) in signals.iter().enumerate() {
        groups.entry((s.symbol.clone(), s.timeframe, s.timestamp)).or_default().push(i);
    }

    for (k, idxs) in groups {
        let Some(&ci) = candle_idx.get(&k) else { continue; };

        // features_json
        let mut fm = Map::new();
        fm.insert("symbol".into(), json!(symbols[ci].to_string()));
        fm.insert("tf".into(), json!(timeframes[ci].as_str()));
        fm.insert("time_ms".into(), json!(cw.timestamps[ci]));
        fm.insert("close".into(), json!(cw.close[ci]));
        fm.insert("high".into(), json!(cw.high[ci]));
        fm.insert("low".into(), json!(cw.low[ci]));
        fm.insert("volume".into(), json!(cw.volume[ci]));

        put_num(&mut fm, "adx", adx[ci]);
        put_num(&mut fm, "atr", atr[ci]);
        put_num(&mut fm, "bb_upper", bb_upper[ci]);
        put_num(&mut fm, "bb_mid", bb_mid[ci]);
        put_num(&mut fm, "bb_lower", bb_lower[ci]);
        put_num(&mut fm, "cci", cci[ci]);

        put_num(&mut fm, "ema20", ema20[ci]);
        put_num(&mut fm, "ema50", ema50[ci]);
        put_num(&mut fm, "ema200", ema200[ci]);

        put_num(&mut fm, "macd", macd[ci]);
        put_num(&mut fm, "macd_signal", macd_signal[ci]);
        put_num(&mut fm, "macd_hist", macd_hist[ci]);

        put_num(&mut fm, "obv", obv[ci]);
        put_num(&mut fm, "rsi", rsi[ci]);
        put_num(&mut fm, "sma", sma[ci]);
        put_num(&mut fm, "stoch_k", stoch_k[ci]);
        put_num(&mut fm, "stoch_d", stoch_d[ci]);
        put_num(&mut fm, "vwap", vwap[ci]);
        put_num(&mut fm, "williams", williams[ci]);

        let features_json = JsonValue::Object(fm);

        // scores_json (все сигналы по свече)
        let mut arr = Vec::with_capacity(idxs.len());
        for &si in &idxs {
            let s = &signals[si];
            arr.push(json!({
                "indicator_id": s.indicator_id,
                "signal_kind": s.signal_kind,
                "signal_sub_id": s.signal_sub_id,
                "side": s.side,
                "score": s.score,
                "value": s.value
            }));
        }
        let scores_json = json!({ "signals": arr });

        for &si in &idxs {
            signals[si].features_json = Some(features_json.clone());
            signals[si].scores_json = Some(scores_json.clone());
            signals[si].candle_is_final = true;
            signals[si].calc_source = 1;
        }
    }
}

fn put_num(m: &mut Map<String, JsonValue>, k: &str, v: f64) {
    if v.is_finite() {
        m.insert(k.to_string(), json!(v));
    }
}