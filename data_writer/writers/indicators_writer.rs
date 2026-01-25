pub struct IndicatorsWriter;

impl IndicatorsWriter {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn write(&self, _data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Write indicators data to DB
        Ok(())
    }
}
