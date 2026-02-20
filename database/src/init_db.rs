use sqlx::{PgPool, Row};
use std::time::Duration;
use tracing::{info, error, warn};

pub struct DatabaseInitializer {
    pool: PgPool,
}

impl DatabaseInitializer {
    pub async fn new(database_url: &str) -> Result<Self, sqlx::Error> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(5)
            .min_connections(1)
            .acquire_timeout(Duration::from_secs(10))
            .idle_timeout(Duration::from_secs(300))
            .connect(database_url)
            .await?;

        Ok(DatabaseInitializer { pool })
    }

    pub async fn initialize_schema(&self) -> Result<(), sqlx::Error> {
        info!("Initializing database schema...");
        
        // Execute the DDL files in order
        let ddl_files = [
            // Core market data tables
            include_str!("../../database/ddl/000_test_data.sql"),
            include_str!("../../database/ddl/001_extensions_and_schemas.sql"),
            include_str!("../../database/ddl/010_market_core.sql"),
            include_str!("../../database/ddl/020_market_candles_tf.sql"),
            include_str!("../../database/ddl/021_market_candles_live.sql"),
            include_str!("../../database/ddl/025_market_staging.sql"),
            include_str!("../../database/ddl/030_market_indicators.sql"),
            include_str!("../../database/ddl/040_market_raw_signals.sql"),
            include_str!("../../database/ddl/045_add_symbol_column.sql"),  // Add symbol column to all tables
            include_str!("../../database/ddl/050_trade_tables.sql"),
            include_str!("../../database/ddl/055_constraints.sql"),
            include_str!("../../database/ddl/056_raw_signals_pk_update.sql"),
            include_str!("../../database/ddl/060_policies.sql"),
            include_str!("../../database/ddl/070_views_all.sql"),
            include_str!("../../database/ddl/080_health_queries.sql"),
            include_str!("../../database/ddl/085_trade_predictors.sql"),
            include_str!("../../database/ddl/090_super_entry_signals.sql"),
            include_str!("../../database/ddl/095_order_manager.sql"),
        ];

        for (index, ddl_content) in ddl_files.iter().enumerate() {
            info!("Executing DDL file #{}...", index);
            if !ddl_content.trim().is_empty() {
                // Execute each DDL file in its own transaction to prevent
                // "current transaction is aborted" cascading failures.
                // If one DDL fails (e.g., hypertable already exists), the next
                // DDL file starts fresh on a clean connection.
                let mut tx = self.pool.begin().await.map_err(|e| {
                    error!("Failed to begin transaction for DDL #{}: {}", index, e);
                    e
                })?;
                
                match sqlx::raw_sql(ddl_content).execute(&mut *tx).await {
                    Ok(_) => {
                        if let Err(e) = tx.commit().await {
                            error!("Failed to commit DDL file #{}: {}", index, e);
                        } else {
                            info!("Successfully executed DDL file #{}", index);
                        }
                    },
                    Err(e) => {
                        warn!("DDL file #{} error (may be safe to ignore if objects already exist): {}", index, e);
                        // Rollback the failed transaction to reset the connection state
                        let _ = tx.rollback().await;
                        // Continue with other files
                        continue;
                    }
                }
            }
        }

        info!("Database schema initialization completed.");
        Ok(())
    }

    pub async fn test_connection(&self) -> Result<(), sqlx::Error> {
        match sqlx::query("SELECT 1").fetch_one(&self.pool).await {
            Ok(row) => {
                let result: i32 = row.get(0);
                if result == 1 {
                    info!("Database connection test successful");
                    Ok(())
                } else {
                    error!("Unexpected ping result: {}", result);
                    Err(sqlx::Error::RowNotFound)
                }
            }
            Err(e) => {
                error!("Database ping failed: {}", e);
                Err(e)
            }
        }
    }

    pub async fn insert_test_data(&self) -> Result<(), sqlx::Error> {
        info!("Inserting test data for integration verification...");
        
        // Insert test record to verify database is fully functional
        let result = sqlx::query(
            "INSERT INTO integration_test_data (symbol, price, volume, timestamp, trade_id) 
             VALUES ($1, $2, $3, $4, $5) 
             ON CONFLICT (trade_id) DO NOTHING"
        )
        .bind("BTCUSDT")
        .bind(45000.0)
        .bind(1.5)
        .bind(chrono::Utc::now().timestamp_millis())
        .bind(999999999i64)
        .execute(&self.pool)
        .await;

        match result {
            Ok(_) => {
                info!("Test data insertion completed successfully");
                Ok(())
            },
            Err(e) => {
                error!("Test data insertion failed: {}", e);
                Err(e)
            }
        }
    }

    pub async fn verify_tables_exist(&self) -> Result<bool, sqlx::Error> {
        // Check if key tables exist
        let table_check_query = "
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'integration_test_data'
            ) as table_exists;";
        
        let row = sqlx::query(table_check_query).fetch_one(&self.pool).await?;
        let table_exists: bool = row.get("table_exists");
        
        if table_exists {
            info!("Verified that integration_test_data table exists");
        } else {
            warn!("integration_test_data table does not exist");
        }
        
        Ok(table_exists)
    }
}

pub async fn initialize_database(database_url: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting database initialization process...");

    let initializer = DatabaseInitializer::new(database_url).await?;

    // Test connection first
    initializer.test_connection().await?;

    // Initialize schema
    initializer.initialize_schema().await?;

    // Verify tables exist
    let tables_exist = initializer.verify_tables_exist().await?;

    if tables_exist {
        // Insert test data
        initializer.insert_test_data().await?;
    } else {
        warn!("Skipping test data insertion as tables don't exist");
    }

    info!("Database initialization process completed successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[tokio::test]
    async fn test_database_initialization() {
        let database_url = env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());
        
        match initialize_database(&database_url).await {
            Ok(_) => println!("Database initialization test passed"),
            Err(e) => println!("Database initialization test failed: {}", e),
        }
    }
}