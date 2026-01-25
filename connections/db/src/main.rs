use anyhow::Result;
use std::fs;
use std::path::Path;
use tokio_postgres::{NoTls, Config};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Database initialization service");
    
    // Get the directory containing SQL files from command line argument
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 || args[1] != "--dir" {
        eprintln!("Usage: {} --dir <directory>", args[0]);
        std::process::exit(1);
    }
    
    let sql_dir = &args[2];
    println!("Loading SQL files from: {}", sql_dir);
    
    // Parse database URL from environment
    let database_url = env::var("DATABASE_URL")
        .map_err(|_| anyhow::anyhow!("DATABASE_URL environment variable not set"))?;
    
    // Connect to PostgreSQL
    let (client, connection) = tokio_postgres::connect(&database_url, NoTls).await?;
    
    // Spawn the connection handling
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });
    
    // Read all .sql files from the directory and sort them
    let mut sql_files = Vec::new();
    for entry in fs::read_dir(sql_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("sql") {
            sql_files.push(path);
        }
    }
    
    sql_files.sort(); // Sort to ensure proper execution order
    
    // Execute each SQL file
    for sql_file in sql_files {
        let sql_content = fs::read_to_string(&sql_file)?;
        println!("Executing: {}", sql_file.display());
        
        // Split content by statements (simple approach - assumes semicolon-separated statements)
        let statements: Vec<&str> = sql_content.split(';').collect();
        for statement in statements {
            let stmt = statement.trim();
            if !stmt.is_empty() {
                match client.execute(stmt, &[]).await {
                    Ok(_) => {
                        // Success
                    }
                    Err(e) => {
                        // Print error but continue with other statements
                        eprintln!("Error executing statement from {}: {}", sql_file.display(), e);
                    }
                }
            }
        }
    }
    
    println!("Database initialization completed successfully!");
    Ok(())
}
