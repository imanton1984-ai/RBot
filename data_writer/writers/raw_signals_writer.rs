pub struct RawSignalsWriter;

impl RawSignalsWriter {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn write(&self, _data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Write raw signals data to DB
        Ok(())
    }
}
