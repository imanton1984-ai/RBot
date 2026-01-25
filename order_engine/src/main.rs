use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Order engine service");
    Ok(())
}
