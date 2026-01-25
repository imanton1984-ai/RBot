pub struct SignalsWriter;

impl SignalsWriter {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn write(&self, _data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Write signals data to DB
        Ok(())
    }
}
