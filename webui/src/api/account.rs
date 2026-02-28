// webui/src/api/account.rs
//
// Account API handlers: spot/futures balances, internal transfers.
//
// Uses Binance Spot API for spot balance and Universal Transfer API
// for moving USDT between Spot ↔ USDⓈ-M Futures.

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info};

use crate::state::AppState;

// ═══════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════

/// Combined spot + futures balance response
#[derive(Debug, Serialize)]
pub struct AccountBalances {
    pub spot_usdt: f64,
    pub spot_available: f64,
    pub futures_usdt: f64,
    pub futures_available: f64,
    pub futures_unrealized_pnl: f64,
    pub total_usdt: f64,
}

/// Transfer request between spot and futures
#[derive(Debug, Deserialize)]
pub struct TransferRequest {
    /// "SPOT_TO_FUTURES" or "FUTURES_TO_SPOT"
    pub direction: String,
    pub amount: f64,
}

/// Transfer response
#[derive(Debug, Serialize)]
pub struct TransferResponse {
    pub success: bool,
    pub tran_id: Option<i64>,
    pub message: String,
}

// ═══════════════════════════════════════════════════════════
// Binance Spot API helpers (reuse HMAC from connections)
// ═══════════════════════════════════════════════════════════

/// HMAC-SHA256 (same implementation as connections/binance_futures.rs)
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    use std::io::Write;

    const BLOCK_SIZE: usize = 64;
    const IPAD: u8 = 0x36;
    const OPAD: u8 = 0x5c;

    let key_prime = if key.len() > BLOCK_SIZE {
        sha256(key).to_vec()
    } else {
        let mut k = key.to_vec();
        k.resize(BLOCK_SIZE, 0);
        k
    };

    let mut key_block = [0u8; BLOCK_SIZE];
    key_block[..key_prime.len().min(BLOCK_SIZE)]
        .copy_from_slice(&key_prime[..key_prime.len().min(BLOCK_SIZE)]);

    let mut inner_data = Vec::with_capacity(BLOCK_SIZE + message.len());
    for i in 0..BLOCK_SIZE {
        inner_data.push(key_block[i] ^ IPAD);
    }
    inner_data.write_all(message).unwrap();
    let inner_hash = sha256(&inner_data);

    let mut outer_data = Vec::with_capacity(BLOCK_SIZE + 32);
    for i in 0..BLOCK_SIZE {
        outer_data.push(key_block[i] ^ OPAD);
    }
    outer_data.write_all(&inner_hash).unwrap();
    sha256(&outer_data)
}

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
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
        result[4 * i..4 * i + 4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

fn sign(secret: &str, query_string: &str) -> String {
    use std::fmt::Write;
    let key = secret.as_bytes();
    let msg = query_string.as_bytes();
    let signature = hmac_sha256(key, msg);
    let mut hex = String::with_capacity(64);
    for byte in &signature {
        write!(hex, "{:02x}", byte).unwrap();
    }
    hex
}

fn timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Load exchange credentials
fn load_credentials() -> Option<(String, String, bool)> {
    let settings = settings::ExchangeSettings::load().ok()?;
    if !settings.has_credentials() {
        return None;
    }
    let testnet = std::env::var("BINANCE_TESTNET")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);
    Some((
        settings.api_key().to_string(),
        settings.api_secret().to_string(),
        testnet,
    ))
}

// ═══════════════════════════════════════════════════════════
// Spot account balance (Binance Spot API)
// ═══════════════════════════════════════════════════════════

/// Binance Spot account balance entry
#[derive(Debug, Deserialize)]
struct SpotBalanceEntry {
    asset: String,
    free: String,
    locked: String,
}

/// Binance Spot account response (partial)
#[derive(Debug, Deserialize)]
struct SpotAccountResponse {
    balances: Vec<SpotBalanceEntry>,
}

/// Get USDT balance from Binance Spot account
async fn get_spot_usdt_balance(
    api_key: &str,
    api_secret: &str,
    testnet: bool,
) -> Result<(f64, f64), String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;

    let base_url = if testnet {
        "https://testnet.binance.vision"
    } else {
        "https://api.binance.com"
    };

    let timestamp = timestamp_ms();
    let query = format!("timestamp={}&recvWindow=10000", timestamp);
    let signature = sign(api_secret, &query);
    let url = format!("{}/api/v3/account?{}&signature={}", base_url, query, signature);

    let resp = client
        .get(&url)
        .header("X-MBX-APIKEY", api_key)
        .send()
        .await
        .map_err(|e| format!("Spot account request failed: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Spot account error: {} — {}", status, body));
    }

    let account: SpotAccountResponse = resp
        .json()
        .await
        .map_err(|e| format!("Spot account parse error: {}", e))?;

    let usdt = account
        .balances
        .iter()
        .find(|b| b.asset == "USDT")
        .map(|b| {
            let free = b.free.parse::<f64>().unwrap_or(0.0);
            let locked = b.locked.parse::<f64>().unwrap_or(0.0);
            (free + locked, free)
        })
        .unwrap_or((0.0, 0.0));

    Ok(usdt)
}

// ═══════════════════════════════════════════════════════════
// Universal Transfer (Spot ↔ USDⓈ-M Futures)
// ═══════════════════════════════════════════════════════════

/// Binance Universal Transfer response
#[derive(Debug, Deserialize)]
struct TransferApiResponse {
    #[serde(rename = "tranId")]
    tran_id: i64,
}

/// Execute universal transfer between spot and USDⓈ-M futures
async fn execute_transfer(
    api_key: &str,
    api_secret: &str,
    testnet: bool,
    transfer_type: &str, // MAIN_UMFUTURE or UMFUTURE_MAIN
    amount: f64,
) -> Result<i64, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("HTTP client error: {}", e))?;

    let base_url = if testnet {
        "https://testnet.binance.vision"
    } else {
        "https://api.binance.com"
    };

    let timestamp = timestamp_ms();
    let query = format!(
        "type={}&asset=USDT&amount={:.8}&timestamp={}&recvWindow=10000",
        transfer_type, amount, timestamp
    );
    let signature = sign(api_secret, &query);
    let url = format!(
        "{}/sapi/v1/asset/transfer?{}&signature={}",
        base_url, query, signature
    );

    let resp = client
        .post(&url)
        .header("X-MBX-APIKEY", api_key)
        .send()
        .await
        .map_err(|e| format!("Transfer request failed: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Transfer failed: {} — {}", status, body));
    }

    let transfer: TransferApiResponse = resp
        .json()
        .await
        .map_err(|e| format!("Transfer parse error: {}", e))?;

    Ok(transfer.tran_id)
}

// ═══════════════════════════════════════════════════════════
// API Handlers
// ═══════════════════════════════════════════════════════════

/// GET /api/account/balances — Get USDT balances for spot + futures
pub async fn get_account_balances(
    State(_state): State<AppState>,
) -> Result<Json<AccountBalances>, StatusCode> {
    let (api_key, api_secret, testnet) = load_credentials().ok_or_else(|| {
        error!("No API credentials available for account balances");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Fetch spot and futures balances in parallel
    let spot_fut = get_spot_usdt_balance(&api_key, &api_secret, testnet);
    let futures_fut = async {
        let client = connections_lib::BinanceFuturesClient::new(&api_key, &api_secret, testnet)
            .map_err(|e| format!("Futures client error: {}", e))?;
        let info = client
            .account_info()
            .await
            .map_err(|e| format!("Futures account error: {}", e))?;
        let wallet = info.total_wallet_balance.parse::<f64>().unwrap_or(0.0);
        let available = info.available_balance.parse::<f64>().unwrap_or(0.0);
        let unrealized = info.total_unrealized_profit.parse::<f64>().unwrap_or(0.0);
        Ok::<(f64, f64, f64), String>((wallet, available, unrealized))
    };

    let (spot_result, futures_result) = tokio::join!(spot_fut, futures_fut);

    let (spot_total, spot_available) = spot_result.unwrap_or_else(|e| {
        debug!("Spot balance unavailable: {}", e);
        (0.0, 0.0)
    });

    let (futures_total, futures_available, futures_unrealized) =
        futures_result.unwrap_or_else(|e| {
            debug!("Futures balance unavailable: {}", e);
            (0.0, 0.0, 0.0)
        });

    let total = spot_total + futures_total;

    Ok(Json(AccountBalances {
        spot_usdt: spot_total,
        spot_available,
        futures_usdt: futures_total,
        futures_available,
        futures_unrealized_pnl: futures_unrealized,
        total_usdt: total,
    }))
}

/// POST /api/account/transfer — Transfer USDT between spot and futures
pub async fn transfer_between_accounts(
    State(_state): State<AppState>,
    Json(request): Json<TransferRequest>,
) -> Result<Json<TransferResponse>, StatusCode> {
    let (api_key, api_secret, testnet) = load_credentials().ok_or_else(|| {
        error!("No API credentials available for transfer");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if request.amount <= 0.0 {
        return Ok(Json(TransferResponse {
            success: false,
            tran_id: None,
            message: "Amount must be greater than 0".to_string(),
        }));
    }

    let transfer_type = match request.direction.as_str() {
        "SPOT_TO_FUTURES" => "MAIN_UMFUTURE",
        "FUTURES_TO_SPOT" => "UMFUTURE_MAIN",
        other => {
            return Ok(Json(TransferResponse {
                success: false,
                tran_id: None,
                message: format!(
                    "Invalid direction: {}. Use SPOT_TO_FUTURES or FUTURES_TO_SPOT",
                    other
                ),
            }));
        }
    };

    info!(
        "Transfer initiated: {} USDT {} (Binance type: {})",
        request.amount, request.direction, transfer_type
    );

    match execute_transfer(&api_key, &api_secret, testnet, transfer_type, request.amount).await {
        Ok(tran_id) => {
            info!("Transfer successful: tranId={}", tran_id);
            Ok(Json(TransferResponse {
                success: true,
                tran_id: Some(tran_id),
                message: format!(
                    "Successfully transferred {:.2} USDT ({})",
                    request.amount, request.direction
                ),
            }))
        }
        Err(e) => {
            error!("Transfer failed: {}", e);
            Ok(Json(TransferResponse {
                success: false,
                tran_id: None,
                message: e,
            }))
        }
    }
}
