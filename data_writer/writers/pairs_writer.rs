pub struct PairsWriter;

impl PairsWriter {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn write(&self, _data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Write pairs data to DB
        Ok(())
    }
}
