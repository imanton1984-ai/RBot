use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionScore {
    pub raw_output: f64,    // Выход модели (цена или вероятность)
    pub confidence: f64,    // Итоговая уверенность (0.0 - 1.0)
    pub source: String,     // "ml_xgboost", "heuristic_ema"
}

impl PredictionScore {
    /// Вычисляет скор для ML-модели (регрессия или классификация)
    pub fn from_ml(output: f64, model_accuracy_metric: f64) -> Self {
        // Если модель исторически точна (metric), уверенность выше.
        // Здесь пока простая логика, которую можно усложнить.
        let confidence = model_accuracy_metric.clamp(0.1, 0.95);
        
        Self {
            raw_output: output,
            confidence,
            source: "ml_onnx".to_string(),
        }
    }

    /// Вычисляет скор для эвристики
    pub fn from_heuristic(price: f64, trend_strength: f64, indicators_match: bool) -> Self {
        let mut conf = 0.5;
        // Если сильный тренд
        if trend_strength.abs() > 0.7 { conf += 0.2; }
        // Если индикаторы подтверждают
        if indicators_match { conf += 0.2; }

        Self {
            raw_output: price,
            confidence: conf.clamp(0.0, 1.0),
            source: "heuristic".to_string(),
        }
    }
}