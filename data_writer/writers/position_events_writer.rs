pub struct PositionEventsWriter;

impl PositionEventsWriter {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn write(&self, _data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Write position events data to DB
        Ok(())
    }
}
