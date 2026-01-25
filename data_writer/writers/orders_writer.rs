pub struct OrdersWriter;

impl OrdersWriter {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn write(&self, _data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Write orders data to DB
        Ok(())
    }
}
