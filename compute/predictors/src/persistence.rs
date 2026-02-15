// compute/predictors/persistence.rs

use anyhow::Result;
use sqlx::{PgPool, Row};
use crate::types::{PredictionRow, PredictorMeta, PredictorId};
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

// NOTE: copy_predictors_binary removed — use BulkPersistor for high-performance writes

pub async fn resolve_symbol_id(pool: &PgPool, symbol: &str) -> Result<i64> {
    let row = sqlx::query("SELECT symbol_id FROM market.pairs WHERE symbol = $1")
        .bind(symbol)
        .fetch_one(pool)
        .await?;
    Ok(row.try_get::<i64, _>("symbol_id")?)
}

pub async fn register_predictor_if_missing(pool: &PgPool, predictor_meta: &PredictorMeta) -> Result<PredictorId> {
    // Check if predictor already exists
    let row = sqlx::query(
        r#"
        SELECT predictor_id
        FROM trade.predictor_registry
        WHERE calc_source = $1 AND name = $2 AND version = $3
        "#,
    )
    .bind(predictor_meta.calc_source.as_int() as i16)
    .bind(&predictor_meta.name)
    .bind(&predictor_meta.version)
    .fetch_optional(pool)
    .await?;

    if let Some(row) = row {
        return Ok(PredictorId(row.try_get::<i64, _>("predictor_id")?));
    }

    // Insert new predictor
    let row = sqlx::query(
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
    )
    .bind(predictor_meta.aspect.as_int() as i16)
    .bind(10i32) // Default horizon
    .bind(predictor_meta.calc_source.as_int() as i16)
    .bind(&predictor_meta.framework)
    .bind(&predictor_meta.name)
    .bind(&predictor_meta.version)
    .bind(None::<String>) // code_hash
    .bind(&predictor_meta.artifact_path)
    .bind(None::<String>) // artifact_sha256
    .bind(&predictor_meta.feature_schema_id)
    .bind(None::<serde_json::Value>) // calibration_json
    .bind(None::<serde_json::Value>) // metrics_json
    .bind(None::<chrono::DateTime<chrono::Utc>>) // trained_from
    .bind(None::<chrono::DateTime<chrono::Utc>>) // trained_to
    .bind(true) // is_active
    .fetch_one(pool)
    .await?;

    Ok(PredictorId(row.try_get::<i64, _>("predictor_id")?))
}

pub async fn get_predictor_meta(pool: &PgPool, predictor_id: i64) -> Result<Option<PredictorMeta>> {
    let row = sqlx::query(
        r#"
        SELECT aspect, calc_source, name, version, framework, artifact_path, feature_schema_id
        FROM trade.predictor_registry
        WHERE predictor_id = $1 AND is_active = true
        "#,
    )
    .bind(predictor_id)
    .fetch_optional(pool)
    .await?;

    if let Some(row) = row {
        let aspect_i16: i16 = row.try_get("aspect")?;
        let calc_source_i16: i16 = row.try_get("calc_source")?;
        Ok(Some(PredictorMeta {
            predictor_id,
            name: row.try_get("name")?,
            version: row.try_get("version")?,
            aspect: crate::types::PredictionAspect::from_int(aspect_i16)
                .ok_or_else(|| anyhow::anyhow!("Invalid aspect value: {}", aspect_i16))?,
            calc_source: crate::types::CalcSource::from_int(calc_source_i16)
                .ok_or_else(|| anyhow::anyhow!("Invalid calc_source value: {}", calc_source_i16))?,
            framework: row.try_get("framework")?,
            artifact_path: row.try_get("artifact_path")?,
            feature_schema_id: row.try_get("feature_schema_id")?,
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
    let rows = sqlx::query(
        r#"
        SELECT
            time, time_ms, symbol_id, symbol, tf_minutes,
            horizon_bars, aspect, calc_source,
            predictor_id, score_norm, value, value_low, value_high, side,
            level_hash, level_kind, level_price, level_strength, level_distance_atr,
            candle_is_final, event_time_ms, details_json, prediction_key
        FROM trade.predictors
        WHERE symbol_id = $1 AND tf_minutes = $2 AND aspect = $3
          AND score_norm >= $4
        ORDER BY time DESC
        LIMIT $5
        "#,
    )
    .bind(symbol_id)
    .bind(tf_minutes)
    .bind(aspect.as_int() as i16)
    .bind(min_score)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    let mut predictors = Vec::new();
    for row in &rows {
        let aspect_i16: i16 = row.try_get("aspect")?;
        let calc_source_i16: i16 = row.try_get("calc_source")?;
        let details: Option<serde_json::Value> = row.try_get("details_json")?;
        predictors.push(PredictionRow {
            time: row.try_get("time")?,
            time_ms: row.try_get("time_ms")?,
            symbol_id: row.try_get("symbol_id")?,
            symbol: row.try_get("symbol")?,
            tf_minutes: row.try_get("tf_minutes")?,
            horizon_bars: row.try_get("horizon_bars")?,
            aspect: crate::types::PredictionAspect::from_int(aspect_i16)
                .ok_or_else(|| anyhow::anyhow!("Invalid aspect value: {}", aspect_i16))?,
            calc_source: crate::types::CalcSource::from_int(calc_source_i16)
                .ok_or_else(|| anyhow::anyhow!("Invalid calc_source value: {}", calc_source_i16))?,
            predictor_id: row.try_get("predictor_id")?,
            score_norm: row.try_get("score_norm")?,
            value: row.try_get("value")?,
            value_low: row.try_get("value_low")?,
            value_high: row.try_get("value_high")?,
            side: row.try_get("side")?,
            level_hash: row.try_get("level_hash")?,
            level_kind: row.try_get("level_kind")?,
            level_price: row.try_get("level_price")?,
            level_strength: row.try_get("level_strength")?,
            level_distance_atr: row.try_get("level_distance_atr")?,
            candle_is_final: row.try_get("candle_is_final")?,
            event_time_ms: row.try_get("event_time_ms")?,
            details_json: details,
            prediction_key: row.try_get("prediction_key")?,
        });
    }

    Ok(predictors)
}

