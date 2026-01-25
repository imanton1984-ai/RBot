pub struct MlDataWriter;

impl MlDataWriter {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn write(&self, _data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Write ML data to DB
        Ok(())
    }
}
