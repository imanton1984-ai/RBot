use std::sync::Arc;
use ort::session::Session;
use dashmap::DashMap;

pub struct ModelPool {
    sessions: DashMap<String, Arc<Session>>,
}

impl ModelPool {
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
        }
    }

    pub fn get_or_load(&self, path: &str, _use_cuda: bool) -> anyhow::Result<Arc<Session>> {
        if let Some(s) = self.sessions.get(path) {
            return Ok(s.clone());
        }

        let session = Session::builder()?.commit_from_file(path)?;
        let shared = Arc::new(session);
        self.sessions.insert(path.to_string(), shared.clone());
        Ok(shared)
    }
}

impl Default for ModelPool {
    fn default() -> Self {
        Self::new()
    }
}
