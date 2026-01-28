use anyhow::Result;
use sqlx::{PgPool, Row};
use std::time::Duration;
use tracing::{debug, error};

#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connect_timeout: Duration,
    pub idle_timeout: Duration,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        DatabaseConfig {
            url: std::env::var("DATABASE_URL").unwrap_or_else(|_| 
                "postgres://postgres:postgres@127.0.0.1:5433/timescaledb_binance".to_string()
            ),
            max_connections: 20,
            min_connections: 5,
            connect_timeout: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(300),
        }
    }
}

pub struct DatabaseConnection {
    pool: PgPool,
    #[allow(dead_code)]
    config: DatabaseConfig,
}

impl DatabaseConnection {
    pub async fn new(config: DatabaseConfig) -> Result<Self> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .acquire_timeout(config.connect_timeout)
            .idle_timeout(config.idle_timeout)
            .connect(&config.url)
            .await?;

        Ok(DatabaseConnection { pool, config })
    }

    pub async fn connect() -> Result<Self> {
        let config = DatabaseConfig::default();
        Self::new(config).await
    }

    pub async fn ping(&self) -> Result<()> {
        match sqlx::query("SELECT 1").fetch_one(&self.pool).await {
            Ok(row) => {
                let result: i32 = row.try_get(0)?;
                if result == 1 {
                    debug!("DB ping ok");
                    Ok(())
                } else {
                    error!("Unexpected ping result: {}", result);
                    anyhow::bail!("Unexpected ping result: {}", result);
                }
            }
            Err(e) => {
                error!("Database ping failed: {}", e);
                anyhow::bail!("Database ping failed: {}", e);
            }
        }
    }

    pub async fn health_check(&self) -> Result<bool> {
        match self.ping().await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    pub fn get_pool(&self) -> &PgPool {
        &self.pool
    }

    pub async fn execute_query<T, F>(&self, query: &str, mapper: F) -> Result<Vec<T>>
    where
        F: Fn(&sqlx::postgres::PgRow) -> T,
    {
        let rows = sqlx::query(query).fetch_all(&self.pool).await?;
        let results: Vec<T> = rows.iter().map(mapper).collect();
        debug!("Executed query returning {} rows", results.len());
        Ok(results)
    }

    pub async fn execute(&self, query: &str) -> Result<u64> {
        let result = sqlx::query(query).execute(&self.pool).await?;
        let rows_affected = result.rows_affected();
        debug!("Executed query affecting {} rows", rows_affected);
        Ok(rows_affected)
    }

    pub async fn test_connection_params(&self) -> Result<()> {
        // Test basic connectivity and common operations
        self.ping().await?;
        
        // Test a simple query
        let version: (String,) = sqlx::query_as("SELECT version()")
            .fetch_one(&self.pool)
            .await?;
        debug!("Database version: {}", version.0);
        
        // Test timezone
        let timezone: (String,) = sqlx::query_as("SELECT current_setting('timezone')")
            .fetch_one(&self.pool)
            .await?;
        debug!("Database timezone: {}", timezone.0);
        
        debug!("Database connection parameters test passed");
        Ok(())
    }
}

// Helper function to initialize database connection
pub async fn init_db_connection() -> Result<DatabaseConnection> {
    let db_conn = DatabaseConnection::connect().await?;
    
    // Run basic health checks
    db_conn.health_check().await?;
    db_conn.test_connection_params().await?;
    
    Ok(db_conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_database_connection() {
        let config = DatabaseConfig {
            url: std::env::var("DATABASE_URL").unwrap_or_else(|_| 
                "postgres://postgres:postgres@127.0.0.1:5433/timescaledb_binance".to_string()
            ),
            max_connections: 5,
            min_connections: 1,
            connect_timeout: Duration::from_secs(5),
            idle_timeout: Duration::from_secs(60),
        };

        match DatabaseConnection::new(config).await {
            Ok(conn) => {
                // Test the connection
                match conn.ping().await {
                    Ok(_) => println!("Database connection test successful"),
                    Err(e) => println!("Database connection failed: {}", e),
                }
            }
            Err(e) => println!("Failed to create database connection: {}", e),
        }
    }
}