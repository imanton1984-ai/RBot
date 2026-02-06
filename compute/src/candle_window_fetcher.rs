use common::{Symbol, Timeframe};
use sqlx::{PgPool, Row};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CandleWindow {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub timestamps: Vec<i64>,
}

pub struct CandleWindowFetcher {
    db_pool: PgPool,
}

impl CandleWindowFetcher {
    pub fn get_db_pool(&self) -> &PgPool {
        &self.db_pool
    }
}

impl CandleWindowFetcher {
    pub fn new(db_pool: PgPool) -> Self {
        Self {
            db_pool,
        }
    }

    pub async fn fetch_candle_window(
        &self,
        symbol: &Symbol,
        timeframe: Timeframe,
        start_time: i64,
        end_time: i64,
    ) -> Result<CandleWindow, Box<dyn std::error::Error + Send + Sync>> {
        let table_name = format!("market.candles_{}", timeframe.as_str());

        let query = format!(
            "SELECT time_ms, open, high, low, close, volume
             FROM {}
             WHERE symbol_id = (SELECT symbol_id FROM market.pairs WHERE symbol = $1)
             AND time_ms >= $2 AND time_ms <= $3
             ORDER BY time_ms ASC",
            table_name
        );

        let rows = sqlx::query(&query)
            .bind(symbol.as_str())
            .bind(start_time)
            .bind(end_time)
            .fetch_all(&self.db_pool)
            .await?;

        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();
        let mut timestamps = Vec::new();

        for row in rows {
            timestamps.push(row.get("time_ms"));
            open.push(row.get::<f64, _>("open"));
            high.push(row.get::<f64, _>("high"));
            low.push(row.get::<f64, _>("low"));
            close.push(row.get::<f64, _>("close"));
            volume.push(row.get::<f64, _>("volume"));
        }

        Ok(CandleWindow {
            open,
            high,
            low,
            close,
            volume,
            timestamps,
        })
    }

    pub async fn fetch_bulk_candles(
        &self,
        symbols: &[Symbol],
        timeframe: Timeframe,
        start_time: i64,
        end_time: i64,
    ) -> Result<std::collections::HashMap<Symbol, CandleWindow>, Box<dyn std::error::Error + Send + Sync>> {
        let mut results = std::collections::HashMap::new();

        for symbol in symbols {
            match self.fetch_candle_window(symbol, timeframe, start_time, end_time).await {
                Ok(window) => {
                    // Only add to results if the window has data
                    if !window.close.is_empty() {
                        results.insert(symbol.clone(), window);
                    }
                },
                Err(e) => {
                    eprintln!("Warning: Failed to fetch candle window for {}: {}", symbol, e);
                    // Continue with other symbols instead of failing the entire operation
                }
            }
        }

        Ok(results)
    }

    pub async fn build_batch_tensors(
        &self,
        windows: std::collections::HashMap<Symbol, CandleWindow>,
        timeframe: common::Timeframe,  // Pass the actual timeframe instead of using placeholder
    ) -> Result<crate::BatchTensor, Box<dyn std::error::Error + Send + Sync>> {
        let mut close = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut volume = Vec::new();
        let mut timestamps = Vec::new();
        let mut symbol_ids = Vec::new();

        for (symbol, window) in windows {
            // Get the symbol_id from the database
            let symbol_id = self.get_symbol_id(&symbol).await?;
            
            // Store the length before moving the vectors
            let window_len = window.close.len();

            // Extend vectors with data for this symbol
            close.extend(window.close);
            high.extend(window.high);
            low.extend(window.low);
            volume.extend(window.volume);
            timestamps.extend(window.timestamps);

            // Add symbol_id for each data point
            for _ in 0..window_len {
                symbol_ids.push(symbol_id as u32);
            }
        }

        Ok(crate::BatchTensor {
            close,
            high,
            low,
            volume,
            timestamps,
            symbol_ids,
            timeframe,
        })
    }
    
    async fn get_symbol_id(&self, symbol: &Symbol) -> Result<i64, Box<dyn std::error::Error + Send + Sync>> {
        let row = sqlx::query("SELECT symbol_id FROM market.pairs WHERE symbol = $1")
            .bind(symbol.as_str())
            .fetch_one(self.get_db_pool())
            .await?;
            
        Ok(row.get("symbol_id"))
    }
}