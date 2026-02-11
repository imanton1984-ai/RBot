// compute/predictors/persistence.rs

use anyhow::Result;
use sqlx::PgPool;
use crate::types::{PredictionRow, PredictorMeta, PredictorId};
use serde_json::Value;

use sqlx::types::Json;

pub async fn upsert_predictors(pool: &PgPool, predictors: Vec<PredictionRow>) -> Result<()> {
    if predictors.is_empty() {
        return Ok(());
    }

    // Prepare the data for bulk insert - using Option types to preserve NULL values
    let mut prediction_ids = Vec::new();
    let mut times = Vec::new();
    let mut time_mss = Vec::new();
    let mut symbol_ids = Vec::new();
    let mut symbols = Vec::new();
    let mut tf_minutes = Vec::new();
    let mut horizon_bars = Vec::new();
    let mut aspects = Vec::new();
    let mut calc_sources = Vec::new();
    let mut predictor_ids = Vec::new();
    let mut score_norms = Vec::new();
    let mut values = Vec::new();
    
    let mut value_lows: Vec<Option<f64>> = Vec::new();
    let mut value_highs: Vec<Option<f64>> = Vec::new();
    let mut sides: Vec<Option<i16>> = Vec::new();
    
    let mut level_hashes: Vec<Option<String>> = Vec::new();
    let mut level_kinds: Vec<Option<i16>> = Vec::new();
    let mut level_prices: Vec<Option<f64>> = Vec::new();
    let mut level_strengths: Vec<Option<f32>> = Vec::new();
    let mut level_distances_atr: Vec<Option<f32>> = Vec::new();
    
    let mut candle_is_finals = Vec::new();
    let mut event_time_mss: Vec<Option<i64>> = Vec::new();
    let mut details_jsons: Vec<Option<Json<serde_json::Value>>> = Vec::new();
    let mut prediction_keys = Vec::new();

    for pred in predictors {
        prediction_ids.push(None::<i64>); // Auto-generated
        times.push(pred.time);
        time_mss.push(pred.time_ms);
        symbol_ids.push(pred.symbol_id);
        symbols.push(pred.symbol);
        tf_minutes.push(pred.tf_minutes);
        horizon_bars.push(pred.horizon_bars);
        aspects.push(pred.aspect.as_int());
        calc_sources.push(pred.calc_source.as_int());
        predictor_ids.push(pred.predictor_id);
        score_norms.push(pred.score_norm);
        values.push(pred.value);
        
        // Preserve Option values without unwrapping
        value_lows.push(pred.value_low);
        value_highs.push(pred.value_high);
        sides.push(pred.side);
        
        level_hashes.push(pred.level_hash);
        level_kinds.push(pred.level_kind);
        level_prices.push(pred.level_price);
        level_strengths.push(pred.level_strength);
        level_distances_atr.push(pred.level_distance_atr);
        
        candle_is_finals.push(pred.candle_is_final);
        event_time_mss.push(pred.event_time_ms);
        details_jsons.push(pred.details_json.map(Json));
        prediction_keys.push(pred.prediction_key);
    }

    // Perform bulk upsert using query + bind to handle Vec<Option<T>>
    let sql = r#"
        INSERT INTO trade.predictors (
            time, time_ms, symbol_id, symbol, tf_minutes,
            horizon_bars, aspect, calc_source, predictor_id,
            score_norm, value, value_low, value_high, side,
            level_hash, level_kind, level_price, level_strength, level_distance_atr,
            candle_is_final, event_time_ms, details_json, prediction_key
        )
        SELECT
            UNNEST($1::TIMESTAMPTZ[]),
            UNNEST($2::BIGINT[]),
            UNNEST($3::BIGINT[]),
            UNNEST($4::TEXT[]),
            UNNEST($5::INT[]),
            UNNEST($6::INT[]),
            UNNEST($7::SMALLINT[]),
            UNNEST($8::SMALLINT[]),
            UNNEST($9::BIGINT[]),
            UNNEST($10::REAL[]),
            UNNEST($11::DOUBLE PRECISION[]),

            UNNEST($12::DOUBLE PRECISION[]),
            UNNEST($13::DOUBLE PRECISION[]),
            UNNEST($14::SMALLINT[]),

            UNNEST($15::TEXT[]),
            UNNEST($16::SMALLINT[]),
            UNNEST($17::DOUBLE PRECISION[]),
            UNNEST($18::REAL[]),
            UNNEST($19::REAL[]),

            UNNEST($20::BOOLEAN[]),
            UNNEST($21::BIGINT[]),
            UNNEST($22::JSONB[]),
            UNNEST($23::TEXT[])
        ON CONFLICT (symbol_id, tf_minutes, time, prediction_key, predictor_id)
        DO UPDATE SET
            score_norm = EXCLUDED.score_norm,
            value = EXCLUDED.value,
            value_low = EXCLUDED.value_low,
            value_high = EXCLUDED.value_high,
            side = EXCLUDED.side,
            level_hash = EXCLUDED.level_hash,
            level_kind = EXCLUDED.level_kind,
            level_price = EXCLUDED.level_price,
            level_strength = EXCLUDED.level_strength,
            level_distance_atr = EXCLUDED.level_distance_atr,
            candle_is_final = EXCLUDED.candle_is_final,
            event_time_ms = EXCLUDED.event_time_ms,
            details_json = EXCLUDED.details_json
    "#;

    sqlx::query(sql)
        .bind(&times)
        .bind(&time_mss)
        .bind(&symbol_ids)
        .bind(&symbols)
        .bind(&tf_minutes)
        .bind(&horizon_bars)
        .bind(&aspects)
        .bind(&calc_sources)
        .bind(&predictor_ids)
        .bind(&score_norms)
        .bind(&values)

        // Bind the optional fields as Vec<Option<_>>
        .bind(&value_lows)
        .bind(&value_highs)
        .bind(&sides)

        .bind(&level_hashes)
        .bind(&level_kinds)
        .bind(&level_prices)
        .bind(&level_strengths)
        .bind(&level_distances_atr)

        .bind(&candle_is_finals)
        .bind(&event_time_mss)
        .bind(&details_jsons)
        .bind(&prediction_keys)
        .execute(pool)
        .await?;

    Ok(())
}

pub async fn resolve_symbol_id(pool: &PgPool, symbol: &str) -> Result<i64> {
    let row = sqlx::query!("SELECT symbol_id FROM market.pairs WHERE symbol = $1", symbol)
        .fetch_one(pool)
        .await?;
    Ok(row.symbol_id)
}

pub async fn register_predictor_if_missing(pool: &PgPool, predictor_meta: &PredictorMeta) -> Result<PredictorId> {
    // Check if predictor already exists
    let row = sqlx::query!(
        r#"
        SELECT predictor_id
        FROM trade.predictor_registry
        WHERE calc_source = $1 AND name = $2 AND version = $3
        "#,
        predictor_meta.calc_source.as_int() as i16,
        predictor_meta.name,
        predictor_meta.version
    )
    .fetch_optional(pool)
    .await?;

    if let Some(row) = row {
        return Ok(PredictorId(row.predictor_id));
    }

    // Insert new predictor
    let row = sqlx::query!(
        r#"
        INSERT INTO trade.predictor_registry (
            aspect, horizon_bars, calc_source, framework,
            name, version, code_hash, artifact_path, artifact_sha256,
            feature_schema_id, calibration_json, metrics_json,
            trained_from, trained_to, is_active
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        RETURNING predictor_id
        "#,
        predictor_meta.aspect.as_int() as i16,
        10i32, // Default horizon
        predictor_meta.calc_source.as_int() as i16,
        predictor_meta.framework,
        predictor_meta.name,
        predictor_meta.version,
        None::<String>, // code_hash
        predictor_meta.artifact_path,
        None::<String>, // artifact_sha256
        predictor_meta.feature_schema_id,
        None::<Value>, // calibration_json
        None::<Value>, // metrics_json
        None::<chrono::DateTime<chrono::Utc>>, // trained_from
        None::<chrono::DateTime<chrono::Utc>>, // trained_to
        true // is_active
    )
    .fetch_one(pool)
    .await?;

    Ok(PredictorId(row.predictor_id))
}

pub async fn get_predictor_meta(pool: &PgPool, predictor_id: i64) -> Result<Option<PredictorMeta>> {
    let row = sqlx::query!(
        r#"
        SELECT aspect, calc_source, name, version, framework, artifact_path, feature_schema_id
        FROM trade.predictor_registry
        WHERE predictor_id = $1 AND is_active = true
        "#,
        predictor_id
    )
    .fetch_optional(pool)
    .await?;

    if let Some(row) = row {
        Ok(Some(PredictorMeta {
            predictor_id,
            name: row.name,
            version: row.version,
            aspect: crate::types::PredictionAspect::from_int(row.aspect)
                .ok_or_else(|| anyhow::anyhow!("Invalid aspect value: {}", row.aspect))?,
            calc_source: crate::types::CalcSource::from_int(row.calc_source)
                .ok_or_else(|| anyhow::anyhow!("Invalid calc_source value: {}", row.calc_source))?,
            framework: row.framework,
            artifact_path: row.artifact_path,
            feature_schema_id: row.feature_schema_id,
        }))
    } else {
        Ok(None)
    }
}

pub async fn get_recent_predictors(
    pool: &PgPool,
    symbol_id: i64,
    tf_minutes: i32,
    aspect: crate::types::PredictionAspect,
    min_score: f32,
    limit: i64,
) -> Result<Vec<PredictionRow>> {
    let rows = sqlx::query!(
        r#"
        SELECT
            prediction_id, time, time_ms, symbol_id, symbol, tf_minutes,
            horizon_bars, aspect as "aspect: i16", calc_source as "calc_source: i16",
            predictor_id, score_norm, value, value_low, value_high, side,
            level_hash, level_kind, level_price, level_strength, level_distance_atr,
            candle_is_final, event_time_ms, details_json::jsonb as "details_json: Json<serde_json::Value>", prediction_key
        FROM trade.predictors
        WHERE symbol_id = $1 AND tf_minutes = $2 AND aspect = $3
          AND score_norm >= $4
        ORDER BY time DESC
        LIMIT $5
        "#,
        symbol_id,
        tf_minutes,
        aspect.as_int() as i16,
        min_score,
        limit
    )
    .fetch_all(pool)
    .await?;

    let mut predictors = Vec::new();
    for row in rows {
        predictors.push(PredictionRow {
            time: row.time,
            time_ms: row.time_ms,
            symbol_id: row.symbol_id,
            symbol: row.symbol,
            tf_minutes: row.tf_minutes,
            horizon_bars: row.horizon_bars,
            aspect: crate::types::PredictionAspect::from_int(row.aspect)
                .ok_or_else(|| anyhow::anyhow!("Invalid aspect value: {}", row.aspect))?,
            calc_source: crate::types::CalcSource::from_int(row.calc_source)
                .ok_or_else(|| anyhow::anyhow!("Invalid calc_source value: {}", row.calc_source))?,
            predictor_id: row.predictor_id,
            score_norm: row.score_norm,
            value: row.value,
            value_low: row.value_low,
            value_high: row.value_high,
            side: row.side,
            level_hash: row.level_hash,
            level_kind: row.level_kind,
            level_price: row.level_price,
            level_strength: row.level_strength,
            level_distance_atr: row.level_distance_atr,
            candle_is_final: row.candle_is_final,
            event_time_ms: row.event_time_ms,
            details_json: row.details_json.map(|json_val| json_val.0), // Unwrap Json wrapper
            prediction_key: row.prediction_key,
        });
    }

    Ok(predictors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_upsert_predictors() {
        // This test would require a database connection
        // For now, just verifying the function signature compiles
    }
}