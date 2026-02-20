// order_manager/src/exchange_info.rs
//
// Exchange Info Cache — кэширует precision/stepSize/tickSize для каждого символа.
// Загружается один раз при старте order_manager, обновляется раз в час.
//
// Используется order_executor для правильного округления quantity и price
// перед отправкой ордера на Binance.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

use connections_lib::{BinanceFuturesClient, SymbolInfo};

/// Кэш Exchange Info — потокобезопасный, обновляемый.
#[derive(Clone)]
pub struct ExchangeInfoCache {
    /// symbol → SymbolInfo
    cache: Arc<RwLock<HashMap<String, SymbolInfo>>>,
    client: BinanceFuturesClient,
}

impl ExchangeInfoCache {
    pub fn new(client: BinanceFuturesClient) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            client,
        }
    }

    /// Загрузить Exchange Info с биржи и закэшировать.
    /// Вызывать при старте и раз в час.
    pub async fn refresh(&self) -> Result<()> {
        let exchange_info = self.client.get_exchange_info().await?;
        let mut cache = self.cache.write().await;
        cache.clear();

        for sym in exchange_info.symbols {
            if sym.status == "TRADING" {
                cache.insert(sym.symbol.clone(), sym);
            }
        }

        info!(
            "ExchangeInfoCache: loaded {} active symbols",
            cache.len()
        );

        Ok(())
    }

    /// Получить SymbolInfo для символа (из кэша).
    pub async fn get(&self, symbol: &str) -> Option<SymbolInfo> {
        let cache = self.cache.read().await;
        cache.get(symbol).cloned()
    }

    /// Округлить quantity по правилам биржи.
    /// Если символ не найден в кэше — fallback на 8 знаков.
    pub async fn round_quantity(&self, symbol: &str, qty: f64) -> f64 {
        let cache = self.cache.read().await;
        match cache.get(symbol) {
            Some(info) => {
                let rounded = info.round_quantity(qty);
                let min = info.lot_min_qty();
                if rounded < min {
                    debug!(
                        "ExchangeInfo: {} qty {:.8} below minQty {:.8}",
                        symbol, rounded, min
                    );
                    return 0.0;
                }
                rounded
            }
            None => {
                warn!(
                    "ExchangeInfo: {} not in cache, using 8-decimal fallback",
                    symbol
                );
                (qty * 1e8).floor() / 1e8
            }
        }
    }

    /// Округлить цену по правилам биржи.
    pub async fn round_price(&self, symbol: &str, price: f64) -> f64 {
        let cache = self.cache.read().await;
        match cache.get(symbol) {
            Some(info) => info.round_price(price),
            None => {
                // Fallback: 2 знака для price
                (price * 100.0).round() / 100.0
            }
        }
    }

    /// Проверить, достаточен ли notional (qty * price >= minNotional).
    pub async fn check_min_notional(&self, symbol: &str, qty: f64, price: f64) -> bool {
        let cache = self.cache.read().await;
        match cache.get(symbol) {
            Some(info) => {
                let notional = qty * price;
                notional >= info.min_notional()
            }
            None => qty * price >= 5.0, // Binance default minimum
        }
    }

    /// Количество символов в кэше
    pub async fn len(&self) -> usize {
        self.cache.read().await.len()
    }

    /// Пуст ли кэш
    pub async fn is_empty(&self) -> bool {
        self.cache.read().await.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantity_rounding_fallback() {
        // Without cache, fallback to 8 decimals
        let qty = 0.123456789_f64;
        let rounded = (qty * 1e8).floor() / 1e8;
        assert!((rounded - 0.12345678_f64).abs() < 1e-10);
    }

    #[test]
    fn test_price_rounding_fallback() {
        let price = 50123.456_f64;
        let rounded = (price * 100.0).round() / 100.0;
        assert!((rounded - 50123.46_f64).abs() < 1e-10);
    }
}
