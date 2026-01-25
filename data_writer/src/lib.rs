pub mod messages;

pub struct DataWriterRouter;

impl DataWriterRouter {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn route_write(&self, _target: &str, _data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}
