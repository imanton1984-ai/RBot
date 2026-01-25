use anyhow::Result;
use sqlx::{Pool, Postgres, Row};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, error};

pub struct DatabaseManager {
    pool: Pool<Postgres>,
}

impl DatabaseManager {
    pub async fn new(connection_string: &str) -> Result<Self> {
        let pool = Pool::connect(connection_string).await?;
        let _ = pool.acquire().await?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &Pool<Postgres> {
        &self.pool
    }

    pub async fn health_check(&self) -> Result<bool> {
        match timeout(Duration::from_secs(5), self.pool.acquire()).await {
            Ok(Ok(mut conn)) => {
                match sqlx::query("SELECT 1 as val").fetch_one(&mut *conn).await {
                    Ok(row) => {
                        let v: i32 = row.get("val");
                        debug!("db health ok: {v}");
                        Ok(v == 1)
                    }
                    Err(e) => { error!("db query failed: {e}"); Ok(false) }
                }
            }
            Ok(Err(e)) => { error!("db acquire failed: {e}"); Ok(false) }
            Err(_) => { error!("db health timeout"); Ok(false) }
        }
    }
}
