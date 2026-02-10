use serde::Deserialize;
use crate::predictors::feature_view::FeatureView;

#[derive(Deserialize, Debug, Clone)]
pub struct FeatureSchema {
    pub schema_id: String,
    pub features: Vec<String>,
}

impl FeatureSchema {
    pub fn from_json_file(path: &str) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let schema = serde_json::from_reader(reader)?;
        Ok(schema)
    }

    pub fn build_vector(&self, view: &FeatureView) -> Vec<f32> {
        self.features.iter().map(|name| view.get_value(name)).collect()
    }
}
