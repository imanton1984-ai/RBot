pub struct CandlesWriter;

impl CandlesWriter {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn write(&self, _data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Write candles data to DB
        Ok(())
    }
}
