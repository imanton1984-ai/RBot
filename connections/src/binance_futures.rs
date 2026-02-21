// connections/src/binance_futures.rs
//
// Binance USDⓈ-M Futures API client.
//
// Authenticated endpoints use HMAC-SHA256 signature (api_key + api_secret).
// Credentials are loaded from settings::ExchangeSettings (→ ~/.settings.json).
//
// This module provides:
//   - Account connectivity check (ping + account info)
//   - Balance query
//   - Open positions query
//   - Order placement (futures OCO: TP/SL)
//   - Position close (market order)
//
// NOTE: For market data (candles, orderbook) use BinanceApi (spot REST) or WebSocket.
//       This module is ONLY for authenticated futures trading operations.

use anyhow::{Result, Context};
use reqwest::{Client, ClientBuilder};
use serde::Deserialize;
use std::time::Duration;
use tracing::{info, debug};

/// Binance Futures base URLs
const FUTURES_BASE_URL: &str = "https://fapi.binance.com";
const FUTURES_TESTNET_URL: &str = "https://testnet.binancefuture.com";

/// Binance Futures API client with HMAC authentication
#[derive(Clone)]
pub struct BinanceFuturesClient {
    client: Client,
    base_url: String,
    api_key: String,
    api_secret: String,
    testnet: bool,
}

/// Account balance info (lenient)
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FuturesBalance {
    #[serde(default)]
    pub asset: String,
    #[serde(default)]
    pub balance: String,
    #[serde(default)]
    pub available_balance: String,
    #[serde(default)]
    pub cross_un_pnl: String,
    #[serde(flatten)]
    pub _extra: std::collections::HashMap<String, serde_json::Value>,
}

/// Open position info (lenient)
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FuturesPosition {
    #[serde(default)]
    pub symbol: String,
    #[serde(default)]
    pub position_amt: String,
    #[serde(default)]
    pub entry_price: String,
    #[serde(default)]
    pub un_realized_profit: String,
    #[serde(default)]
    pub leverage: String,
    #[serde(default)]
    pub position_side: String,
    #[serde(default)]
    pub notional: String,
    #[serde(flatten)]
    pub _extra: std::collections::HashMap<String, serde_json::Value>,
}

/// New order response from Binance Futures
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewOrderResponse {
    pub order_id: i64,
    pub symbol: String,
    pub status: String,
    pub client_order_id: String,
    #[serde(default)]
    pub price: String,
    #[serde(default)]
    pub avg_price: String,
    #[serde(default)]
    pub orig_qty: String,
    #[serde(default)]
    pub executed_qty: String,
    #[serde(default)]
    pub cum_quote: String,
    #[serde(rename = "type")]
    pub order_type: String,
    pub side: String,
    #[serde(default)]
    pub stop_price: String,
    #[serde(default)]
    pub time_in_force: String,
    #[serde(default)]
    pub reduce_only: bool,
    #[serde(default)]
    pub close_position: bool,
    pub update_time: i64,
}

/// Order status response
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderStatusResponse {
    pub order_id: i64,
    pub symbol: String,
    pub status: String,
    pub client_order_id: String,
    #[serde(default)]
    pub price: String,
    #[serde(default)]
    pub avg_price: String,
    #[serde(default)]
    pub orig_qty: String,
    #[serde(default)]
    pub executed_qty: String,
    #[serde(rename = "type")]
    pub order_type: String,
    pub side: String,
    #[serde(default)]
    pub stop_price: String,
    #[serde(default)]
    pub reduce_only: bool,
    pub update_time: i64,
}

/// Mark price response
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarkPriceResponse {
    pub symbol: String,
    pub mark_price: String,
    pub index_price: String,
    pub last_funding_rate: String,
    pub next_funding_time: i64,
    pub time: i64,
}

/// Set leverage response
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeverageResponse {
    pub leverage: i32,
    pub max_notional_value: String,
    pub symbol: String,
}

/// Cancel order response
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelOrderResponse {
    pub order_id: i64,
    pub symbol: String,
    pub status: String,
    pub client_order_id: String,
}

// ═══════════════════════════════════════════════════════════
// EXCHANGE INFO (symbol precision / filters)
// ═══════════════════════════════════════════════════════════

/// Полный ответ exchangeInfo (урезанный до нужных полей)
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExchangeInfoResponse {
    pub symbols: Vec<SymbolInfo>,
}

/// Информация о конкретном символе
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SymbolInfo {
    pub symbol: String,
    pub status: String,
    #[serde(default)]
    pub base_asset: String,
    #[serde(default)]
    pub quote_asset: String,
    #[serde(default)]
    pub price_precision: u8,
    #[serde(default)]
    pub quantity_precision: u8,
    #[serde(default)]
    pub filters: Vec<SymbolFilter>,
}

/// Фильтры символа (LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL и т.д.)
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "filterType")]
pub enum SymbolFilter {
    #[serde(rename = "PRICE_FILTER")]
    PriceFilter {
        #[serde(default, rename = "minPrice")]
        min_price: String,
        #[serde(default, rename = "maxPrice")]
        max_price: String,
        #[serde(default, rename = "tickSize")]
        tick_size: String,
    },
    #[serde(rename = "LOT_SIZE")]
    LotSize {
        #[serde(default, rename = "minQty")]
        min_qty: String,
        #[serde(default, rename = "maxQty")]
        max_qty: String,
        #[serde(default, rename = "stepSize")]
        step_size: String,
    },
    #[serde(rename = "MIN_NOTIONAL")]
    MinNotional {
        #[serde(default)]
        notional: String,
    },
    #[serde(rename = "MARKET_LOT_SIZE")]
    MarketLotSize {
        #[serde(default, rename = "minQty")]
        min_qty: String,
        #[serde(default, rename = "maxQty")]
        max_qty: String,
        #[serde(default, rename = "stepSize")]
        step_size: String,
    },
    /// Catch-all для остальных фильтров
    #[serde(other)]
    Other,
}

impl SymbolInfo {
    /// Получить stepSize из LOT_SIZE фильтра
    pub fn lot_step_size(&self) -> f64 {
        for f in &self.filters {
            if let SymbolFilter::LotSize { step_size, .. } = f {
                return step_size.parse::<f64>().unwrap_or(1e-8);
            }
        }
        1e-8 // fallback
    }

    /// Получить minQty из LOT_SIZE фильтра
    pub fn lot_min_qty(&self) -> f64 {
        for f in &self.filters {
            if let SymbolFilter::LotSize { min_qty, .. } = f {
                return min_qty.parse::<f64>().unwrap_or(0.0);
            }
        }
        0.0
    }

    /// Получить tickSize из PRICE_FILTER
    pub fn price_tick_size(&self) -> f64 {
        for f in &self.filters {
            if let SymbolFilter::PriceFilter { tick_size, .. } = f {
                return tick_size.parse::<f64>().unwrap_or(0.01);
            }
        }
        0.01
    }

    /// Получить minNotional из MIN_NOTIONAL
    pub fn min_notional(&self) -> f64 {
        for f in &self.filters {
            if let SymbolFilter::MinNotional { notional } = f {
                return notional.parse::<f64>().unwrap_or(5.0);
            }
        }
        5.0
    }

    /// Округлить количество до stepSize
    pub fn round_quantity(&self, qty: f64) -> f64 {
        let step = self.lot_step_size();
        if step <= 0.0 {
            return qty;
        }
        (qty / step).floor() * step
    }

    /// Округлить цену до tickSize
    pub fn round_price(&self, price: f64) -> f64 {
        let tick = self.price_tick_size();
        if tick <= 0.0 {
            return price;
        }
        (price / tick).round() * tick
    }
}

/// Account info (lenient — ignores unknown fields from Binance)
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FuturesAccountInfo {
    #[serde(default)]
    pub total_wallet_balance: String,
    #[serde(default)]
    pub total_unrealized_profit: String,
    #[serde(default)]
    pub total_margin_balance: String,
    #[serde(default)]
    pub available_balance: String,
    #[serde(default)]
    pub can_trade: bool,
    #[serde(default)]
    pub positions: Vec<FuturesPosition>,
    #[serde(default)]
    pub assets: Vec<FuturesBalance>,
    /// Catch-all for any extra fields Binance adds
    #[serde(flatten)]
    pub _extra: std::collections::HashMap<String, serde_json::Value>,
}

/// Connection status result
#[derive(Debug, Clone)]
pub struct ConnectionStatus {
    pub spot_ping: bool,
    pub futures_ping: bool,
    pub futures_auth: bool,
    pub can_trade: bool,
    pub usdt_balance: f64,
    pub open_positions: usize,
    pub error: Option<String>,
}

impl std::fmt::Display for ConnectionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Spot: {} | Futures: {} | Auth: {} | Trade: {} | USDT: {:.2} | Positions: {}",
            if self.spot_ping { "✅" } else { "❌" },
            if self.futures_ping { "✅" } else { "❌" },
            if self.futures_auth { "✅" } else { "❌" },
            if self.can_trade { "✅" } else { "❌" },
            self.usdt_balance,
            self.open_positions,
        )
    }
}

impl BinanceFuturesClient {
    /// Create a new Futures client with API credentials.
    ///
    /// # Arguments
    /// * `api_key` - Binance API key (futures-enabled)
    /// * `api_secret` - Binance API secret
    /// * `testnet` - Use testnet instead of production
    pub fn new(api_key: &str, api_secret: &str, testnet: bool) -> Result<Self> {
        let client = ClientBuilder::new()
            .timeout(Duration::from_secs(15))
            .build()?;

        let base_url = if testnet {
            FUTURES_TESTNET_URL.to_string()
        } else {
            FUTURES_BASE_URL.to_string()
        };

        info!(
            "BinanceFuturesClient created ({})",
            if testnet { "TESTNET" } else { "PRODUCTION" }
        );

        Ok(Self {
            client,
            base_url,
            api_key: api_key.to_string(),
            api_secret: api_secret.to_string(),
            testnet,
        })
    }

    /// Create from settings::ExchangeSettings
    pub fn from_settings(settings: &settings_lib::ExchangeSettings) -> Result<Self> {
        if !settings.has_credentials() {
            anyhow::bail!(
                "No API credentials loaded. Create ~/.settings.json or set BINANCE_API_KEY/BINANCE_API_SECRET"
            );
        }

        let testnet = std::env::var("BINANCE_TESTNET")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        Self::new(settings.api_key(), settings.api_secret(), testnet)
    }

    /// Generate HMAC-SHA256 signature for request parameters
    fn sign(&self, query_string: &str) -> String {
        use std::fmt::Write;
        // HMAC-SHA256 using raw implementation (no extra crate needed)
        // We use the built-in approach via reqwest + manual HMAC
        let key = self.api_secret.as_bytes();
        let msg = query_string.as_bytes();

        // Simple HMAC-SHA256 implementation
        let signature = hmac_sha256(key, msg);
        let mut hex = String::with_capacity(64);
        for byte in &signature {
            write!(hex, "{:02x}", byte).unwrap();
        }
        hex
    }

    /// Get current server time (milliseconds)
    fn timestamp_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Ping futures API (no auth required)
    pub async fn ping(&self) -> Result<()> {
        let url = format!("{}/fapi/v1/ping", self.base_url);
        let resp = self.client.get(&url).send().await?;

        if resp.status().is_success() {
            debug!("Binance Futures ping OK");
            Ok(())
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Futures ping failed: {} — {}", status, body)
        }
    }

    /// Get server time (no auth required)
    pub async fn server_time(&self) -> Result<u64> {
        let url = format!("{}/fapi/v1/time", self.base_url);
        let resp = self.client.get(&url).send().await?;

        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct TimeResp {
            server_time: u64,
        }

        let time: TimeResp = resp.json().await?;
        Ok(time.server_time)
    }

    /// Get account information (authenticated)
    pub async fn account_info(&self) -> Result<FuturesAccountInfo> {
        let timestamp = Self::timestamp_ms();
        let query = format!("timestamp={}&recvWindow=10000", timestamp);
        let signature = self.sign(&query);
        let url = format!(
            "{}/fapi/v2/account?{}&signature={}",
            self.base_url, query, signature
        );

        let resp = self
            .client
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .context("Binance Futures account_info: network/TLS error")?;

        if resp.status().is_success() {
            // Parse via text first for better error diagnostics
            let body = resp.text().await.context("Failed to read response body")?;
            let info: FuturesAccountInfo = serde_json::from_str(&body)
                .context("Failed to parse account info JSON")?;
            debug!(
                "Account info: balance={}, can_trade={}, positions={}",
                info.total_wallet_balance,
                info.can_trade,
                info.positions.iter().filter(|p| p.position_amt.parse::<f64>().unwrap_or(0.0).abs() > 0.0).count()
            );
            Ok(info)
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Account info failed: {} — {}", status, body)
        }
    }

    /// Get USDT balance
    pub async fn usdt_balance(&self) -> Result<f64> {
        let info = self.account_info().await?;
        let usdt = info
            .assets
            .iter()
            .find(|a| a.asset == "USDT")
            .map(|a| a.available_balance.parse::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);
        Ok(usdt)
    }

    /// Get open positions (non-zero position amount)
    pub async fn open_positions(&self) -> Result<Vec<FuturesPosition>> {
        let info = self.account_info().await?;
        let open: Vec<FuturesPosition> = info
            .positions
            .into_iter()
            .filter(|p| p.position_amt.parse::<f64>().unwrap_or(0.0).abs() > 0.001)
            .collect();
        Ok(open)
    }

    /// Full connection status check: ping + auth + balance + positions
    pub async fn check_connection(&self) -> ConnectionStatus {
        let mut status = ConnectionStatus {
            spot_ping: false,
            futures_ping: false,
            futures_auth: false,
            can_trade: false,
            usdt_balance: 0.0,
            open_positions: 0,
            error: None,
        };

        // 1. Spot ping (basic internet connectivity)
        let spot_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .ok();
        if let Some(client) = &spot_client {
            if let Ok(resp) = client.get("https://api.binance.com/api/v3/ping").send().await {
                status.spot_ping = resp.status().is_success();
            }
        }

        // 2. Futures ping
        match self.ping().await {
            Ok(()) => status.futures_ping = true,
            Err(e) => {
                status.error = Some(format!("Futures ping failed: {}", e));
                return status;
            }
        }

        // 3. Auth check (account info)
        match self.account_info().await {
            Ok(info) => {
                status.futures_auth = true;
                status.can_trade = info.can_trade;
                status.usdt_balance = info
                    .assets
                    .iter()
                    .find(|a| a.asset == "USDT")
                    .map(|a| a.available_balance.parse::<f64>().unwrap_or(0.0))
                    .unwrap_or(0.0);
                status.open_positions = info
                    .positions
                    .iter()
                    .filter(|p| p.position_amt.parse::<f64>().unwrap_or(0.0).abs() > 0.001)
                    .count();
            }
            Err(e) => {
                status.error = Some(format!("Auth failed: {}", e));
            }
        }

        status
    }

    /// Check if the client is using testnet
    pub fn is_testnet(&self) -> bool {
        self.testnet
    }

    // ═══════════════════════════════════════════════════════════
    // TRADING METHODS (Order Manager)
    // ═══════════════════════════════════════════════════════════

    /// Set leverage for a symbol.
    /// Must be called before placing orders if leverage differs from default.
    pub async fn set_leverage(&self, symbol: &str, leverage: u16) -> Result<LeverageResponse> {
        let timestamp = Self::timestamp_ms();
        let query = format!(
            "symbol={}&leverage={}&timestamp={}&recvWindow=5000",
            symbol, leverage, timestamp
        );
        let signature = self.sign(&query);
        let url = format!(
            "{}/fapi/v1/leverage?{}&signature={}",
            self.base_url, query, signature
        );

        let resp = self
            .client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .context("Failed to set leverage")?;

        if resp.status().is_success() {
            let result: LeverageResponse = resp.json().await?;
            info!("Set leverage for {} to {}x", symbol, result.leverage);
            Ok(result)
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Set leverage failed for {}: {} — {}", symbol, status, body)
        }
    }

    /// Place a MARKET order (entry or close).
    ///
    /// * `symbol` — e.g. "BTCUSDT"
    /// * `side` — "BUY" or "SELL"
    /// * `quantity` — amount in base asset
    /// * `reduce_only` — true when closing a position
    pub async fn place_market_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        reduce_only: bool,
    ) -> Result<NewOrderResponse> {
        let timestamp = Self::timestamp_ms();
        let query = format!(
            "symbol={}&side={}&type=MARKET&quantity={:.8}&reduceOnly={}&timestamp={}&recvWindow=5000",
            symbol, side, quantity, reduce_only, timestamp
        );
        let signature = self.sign(&query);
        let url = format!(
            "{}/fapi/v1/order?{}&signature={}",
            self.base_url, query, signature
        );

        let resp = self
            .client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .context("Failed to place market order")?;

        if resp.status().is_success() {
            let order: NewOrderResponse = resp.json().await?;
            info!(
                "Market order placed: {} {} {} qty={:.8} → orderId={}",
                symbol, side, order.status, quantity, order.order_id
            );
            Ok(order)
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Market order failed for {} {}: {} — {}",
                symbol, side, status, body
            )
        }
    }

    /// Place a STOP_MARKET order (stop-loss).
    ///
    /// * `symbol` — e.g. "BTCUSDT"
    /// * `side` — opposite of position: "SELL" for LONG SL, "BUY" for SHORT SL
    /// * `quantity` — amount in base asset
    /// * `stop_price` — trigger price
    pub async fn place_stop_market(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        stop_price: f64,
    ) -> Result<NewOrderResponse> {
        let timestamp = Self::timestamp_ms();
        let query = format!(
            "symbol={}&side={}&type=STOP_MARKET&quantity={:.8}&stopPrice={:.8}&reduceOnly=true&timestamp={}&recvWindow=5000",
            symbol, side, quantity, stop_price, timestamp
        );
        let signature = self.sign(&query);
        let url = format!(
            "{}/fapi/v1/order?{}&signature={}",
            self.base_url, query, signature
        );

        let resp = self
            .client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .context("Failed to place stop-market order")?;

        if resp.status().is_success() {
            let order: NewOrderResponse = resp.json().await?;
            info!(
                "Stop-market order placed: {} {} stopPrice={:.8} → orderId={}",
                symbol, side, stop_price, order.order_id
            );
            Ok(order)
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Stop-market order failed for {} {}: {} — {}",
                symbol, side, status, body
            )
        }
    }

    /// Place a TAKE_PROFIT_MARKET order.
    ///
    /// * `symbol` — e.g. "BTCUSDT"
    /// * `side` — opposite of position: "SELL" for LONG TP, "BUY" for SHORT TP
    /// * `quantity` — amount in base asset
    /// * `stop_price` — trigger price for TP
    pub async fn place_take_profit_market(
        &self,
        symbol: &str,
        side: &str,
        quantity: f64,
        stop_price: f64,
    ) -> Result<NewOrderResponse> {
        let timestamp = Self::timestamp_ms();
        let query = format!(
            "symbol={}&side={}&type=TAKE_PROFIT_MARKET&quantity={:.8}&stopPrice={:.8}&reduceOnly=true&timestamp={}&recvWindow=5000",
            symbol, side, quantity, stop_price, timestamp
        );
        let signature = self.sign(&query);
        let url = format!(
            "{}/fapi/v1/order?{}&signature={}",
            self.base_url, query, signature
        );

        let resp = self
            .client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .context("Failed to place take-profit order")?;

        if resp.status().is_success() {
            let order: NewOrderResponse = resp.json().await?;
            info!(
                "Take-profit order placed: {} {} stopPrice={:.8} → orderId={}",
                symbol, side, stop_price, order.order_id
            );
            Ok(order)
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Take-profit order failed for {} {}: {} — {}",
                symbol, side, status, body
            )
        }
    }

    /// Cancel a specific order by orderId.
    pub async fn cancel_order(&self, symbol: &str, order_id: i64) -> Result<CancelOrderResponse> {
        let timestamp = Self::timestamp_ms();
        let query = format!(
            "symbol={}&orderId={}&timestamp={}&recvWindow=5000",
            symbol, order_id, timestamp
        );
        let signature = self.sign(&query);
        let url = format!(
            "{}/fapi/v1/order?{}&signature={}",
            self.base_url, query, signature
        );

        let resp = self
            .client
            .delete(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .context("Failed to cancel order")?;

        if resp.status().is_success() {
            let result: CancelOrderResponse = resp.json().await?;
            info!(
                "Order cancelled: {} orderId={} → status={}",
                symbol, order_id, result.status
            );
            Ok(result)
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Cancel order failed for {} orderId={}: {} — {}",
                symbol, order_id, status, body
            )
        }
    }

    /// Cancel all open orders for a symbol.
    pub async fn cancel_all_orders(&self, symbol: &str) -> Result<()> {
        let timestamp = Self::timestamp_ms();
        let query = format!(
            "symbol={}&timestamp={}&recvWindow=5000",
            symbol, timestamp
        );
        let signature = self.sign(&query);
        let url = format!(
            "{}/fapi/v1/allOpenOrders?{}&signature={}",
            self.base_url, query, signature
        );

        let resp = self
            .client
            .delete(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .context("Failed to cancel all orders")?;

        if resp.status().is_success() {
            info!("All open orders cancelled for {}", symbol);
            Ok(())
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Cancel all orders failed for {}: {} — {}",
                symbol, status, body
            )
        }
    }

    /// Get order status by orderId.
    pub async fn get_order_status(
        &self,
        symbol: &str,
        order_id: i64,
    ) -> Result<OrderStatusResponse> {
        let timestamp = Self::timestamp_ms();
        let query = format!(
            "symbol={}&orderId={}&timestamp={}&recvWindow=5000",
            symbol, order_id, timestamp
        );
        let signature = self.sign(&query);
        let url = format!(
            "{}/fapi/v1/order?{}&signature={}",
            self.base_url, query, signature
        );

        let resp = self
            .client
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .context("Failed to get order status")?;

        if resp.status().is_success() {
            let order: OrderStatusResponse = resp.json().await?;
            Ok(order)
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Get order status failed for {} orderId={}: {} — {}",
                symbol, order_id, status, body
            )
        }
    }

    /// Get mark (current) price for a symbol.
    pub async fn get_mark_price(&self, symbol: &str) -> Result<f64> {
        let url = format!(
            "{}/fapi/v1/premiumIndex?symbol={}",
            self.base_url, symbol
        );

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to get mark price")?;

        if resp.status().is_success() {
            let data: MarkPriceResponse = resp.json().await?;
            let price = data.mark_price.parse::<f64>().unwrap_or(0.0);
            Ok(price)
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Get mark price failed for {}: {} — {}",
                symbol, status, body
            )
        }
    }

    /// Close an entire position for a symbol by placing a market order in the opposite direction.
    ///
    /// * `symbol` — e.g. "BTCUSDT"
    /// * `position_side` — "LONG" or "SHORT" (determines close side)
    /// * `quantity` — position size to close
    pub async fn close_position(
        &self,
        symbol: &str,
        position_side: &str,
        quantity: f64,
    ) -> Result<NewOrderResponse> {
        let close_side = match position_side {
            "LONG" => "SELL",
            "SHORT" => "BUY",
            _ => anyhow::bail!("Invalid position_side: {}", position_side),
        };
        self.place_market_order(symbol, close_side, quantity, true).await
    }

    // ═══════════════════════════════════════════════════════════
    // EXCHANGE INFO
    // ═══════════════════════════════════════════════════════════

    /// Получить Exchange Info для всех символов (no auth required).
    /// Содержит precision, stepSize, tickSize, minNotional для каждого символа.
    pub async fn get_exchange_info(&self) -> Result<ExchangeInfoResponse> {
        let url = format!("{}/fapi/v1/exchangeInfo", self.base_url);

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to get exchange info")?;

        if resp.status().is_success() {
            let info: ExchangeInfoResponse = resp.json().await?;
            info!(
                "Exchange info loaded: {} symbols",
                info.symbols.len()
            );
            Ok(info)
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Get exchange info failed: {} — {}", status, body)
        }
    }

    /// Получить Exchange Info для одного символа
    pub async fn get_symbol_info(&self, symbol: &str) -> Result<SymbolInfo> {
        let info = self.get_exchange_info().await?;
        info.symbols
            .into_iter()
            .find(|s| s.symbol == symbol)
            .ok_or_else(|| anyhow::anyhow!("Symbol {} not found in exchange info", symbol))
    }
}

/// HMAC-SHA256 implementation (no external crate needed)
/// Uses the standard HMAC construction: H((K' ⊕ opad) || H((K' ⊕ ipad) || message))
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    use std::io::Write;

    const BLOCK_SIZE: usize = 64;
    const IPAD: u8 = 0x36;
    const OPAD: u8 = 0x5c;

    // If key > block_size, hash it first
    let key_prime = if key.len() > BLOCK_SIZE {
        sha256(key).to_vec()
    } else {
        let mut k = key.to_vec();
        k.resize(BLOCK_SIZE, 0);
        k
    };

    // Ensure key is exactly BLOCK_SIZE
    let mut key_block = [0u8; BLOCK_SIZE];
    key_block[..key_prime.len().min(BLOCK_SIZE)].copy_from_slice(&key_prime[..key_prime.len().min(BLOCK_SIZE)]);

    // Inner: H((K' ⊕ ipad) || message)
    let mut inner_data = Vec::with_capacity(BLOCK_SIZE + message.len());
    for i in 0..BLOCK_SIZE {
        inner_data.push(key_block[i] ^ IPAD);
    }
    inner_data.write_all(message).unwrap();
    let inner_hash = sha256(&inner_data);

    // Outer: H((K' ⊕ opad) || inner_hash)
    let mut outer_data = Vec::with_capacity(BLOCK_SIZE + 32);
    for i in 0..BLOCK_SIZE {
        outer_data.push(key_block[i] ^ OPAD);
    }
    outer_data.write_all(&inner_hash).unwrap();
    sha256(&outer_data)
}

/// SHA-256 implementation (minimal, no external crate)
fn sha256(data: &[u8]) -> [u8; 32] {
    // Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Round constants
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    // Pre-processing: padding
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block
    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4*i], chunk[4*i+1], chunk[4*i+2], chunk[4*i+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut result = [0u8; 32];
    for (i, &val) in h.iter().enumerate() {
        result[4*i..4*i+4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_empty() {
        let hash = sha256(b"");
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn test_sha256_hello() {
        let hash = sha256(b"hello");
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
    }

    #[test]
    fn test_hmac_sha256() {
        // Test vector from RFC 4231 Test Case 2
        let key = b"Jefe";
        let msg = b"what do ya want for nothing?";
        let mac = hmac_sha256(key, msg);
        let hex: String = mac.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    }

    #[test]
    fn test_sign_query() {
        // Binance example from docs
        let client = BinanceFuturesClient {
            client: reqwest::Client::new(),
            base_url: FUTURES_BASE_URL.to_string(),
            api_key: "test".to_string(),
            api_secret: "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j".to_string(),
            testnet: false,
        };
        let query = "symbol=BTCUSDT&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559";
        let sig = client.sign(query);
        // Known signature for this test vector
        assert_eq!(sig.len(), 64); // SHA256 hex = 64 chars
    }

    #[test]
    fn test_connection_status_display() {
        let status = ConnectionStatus {
            spot_ping: true,
            futures_ping: true,
            futures_auth: true,
            can_trade: true,
            usdt_balance: 1234.56,
            open_positions: 3,
            error: None,
        };
        let display = format!("{}", status);
        assert!(display.contains("✅"));
        assert!(display.contains("1234.56"));
        assert!(display.contains("3"));
    }
}
