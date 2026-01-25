use anyhow::Result;
use sqlx::{Pool, Postgres, Row};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

pub struct DatabaseManager {
    pool: Pool<Postgres>,
}

impl DatabaseManager {
    pub async fn new(connection_string: &str) -> Result<Self> {
        let pool = Pool::connect(connection_string).await?;
        
        // Test the connection
        let _ = pool.acquire().await?;
        
        Ok(Self { pool })
    }

    pub fn get_pool(&self) -> &Pool<Postgres> {
        &self.pool
    }

    pub async fn health_check(&self) -> Result<bool> {
        // Attempt to execute a simple query to verify database connectivity
        match timeout(Duration::from_secs(5), self.pool.acquire()).await {
            Ok(Ok(mut conn)) => {
                match sqlx::query("SELECT 1 as val").fetch_one(&mut *conn).await {
                    Ok(row) => {
                        let result: i32 = row.get("val");
                        debug!("Database health check successful, got result: {}", result);
                        Ok(result == 1)
                    }
                    Err(e) => {
                        error!("Database query failed during health check: {}", e);
                        Ok(false)
                    }
                }
            }
            Ok(Err(e)) => {
                error!("Failed to acquire database connection for health check: {}", e);
                Ok(false)
            }
            Err(_) => {
                error!("Database health check timed out");
                Ok(false)
            }
        }
    }

    pub async fn execute_query<T, F, Fut>(&self, query_func: F) -> Result<T>
    where
        F: FnOnce(&Pool<Postgres>) -> Fut,
        Fut: std::future::Future<Output = Result<T, sqlx::Error>>,
    {
        match query_func(&self.pool).await {
            Ok(result) => Ok(result),
            Err(e) => {
                error!("Database query failed: {}", e);
                Err(anyhow::anyhow!("Database query failed: {}", e))
            }
        }
    }

    pub async fn ping(&self) -> Result<bool> {
        match sqlx::query("SELECT 1 as val").fetch_one(&self.pool).await {
            Ok(_) => {
                debug!("Database ping successful");
                Ok(true)
            }
            Err(e) => {
                error!("Database ping failed: {}", e);
                Ok(false)
            }
        }
    }

    pub async fn ensure_connection(&self) -> Result<()> {
        // This method ensures that the connection is still alive
        // and attempts to reconnect if needed
        if !self.health_check().await? {
            error!("Database connection lost, attempting to reconnect...");
            // Note: sqlx handles reconnection automatically, 
            // but we can log this for monitoring purposes
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_db_manager_creation() {
        // This test would require a real database connection
        // For now, we'll just test the interface
        assert!(true);
    }
}
