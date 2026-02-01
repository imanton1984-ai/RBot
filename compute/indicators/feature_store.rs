use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use common::{Symbol, Timeframe};

#[derive(Debug, Clone)]
pub struct FeatureValue {
    pub timestamp: i64,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct FeatureSet {
    pub features: HashMap<String, FeatureValue>,
    pub last_update: i64,
}

pub struct FeatureStore {
    store: Arc<RwLock<HashMap<(Symbol, Timeframe), Arc<FeatureSet>>>>,
}

impl FeatureStore {
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn update_features(
        &self,
        symbol: Symbol,
        timeframe: Timeframe,
        features: HashMap<String, FeatureValue>,
    ) {
        let key = (symbol, timeframe);
        let feature_set = Arc::new(FeatureSet {
            features,
            last_update: chrono::Utc::now().timestamp_millis(),
        });

        let mut store = self.store.write().await;
        store.insert(key, feature_set);
    }

    pub async fn get_features(
        &self,
        symbol: &Symbol,
        timeframe: &Timeframe,
    ) -> Option<Arc<FeatureSet>> {
        let key = (symbol.clone(), *timeframe);
        let store = self.store.read().await;
        store.get(&key).cloned()
    }

    pub async fn has_features(
        &self,
        symbol: &Symbol,
        timeframe: &Timeframe,
    ) -> bool {
        let key = (symbol.clone(), *timeframe);
        let store = self.store.read().await;
        store.contains_key(&key)
    }
}

impl Default for FeatureStore {
    fn default() -> Self {
        Self::new()
    }
}