use anyhow::{anyhow, Context, Result};
use chrono::{TimeZone, Utc};
use common::timeframe::TimeFrame;
use common::{load_config, AppConfig};
use futures::{stream, StreamExt};
use futures::SinkExt;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio::time::{Duration, Instant};
use tokio_postgres::binary_copy::BinaryCopyInWriter;
use tokio_postgres::types::{Type};
use tokio_postgres::NoTls;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

#[derive(Debug, Clone)]
pub struct PairInfo {
    pub symbol_id: i64,
    pub symbol: String, // "BTCUSDT"
}
#[derive(Debug, Clone)]
pub struct CandleRow {
    pub time_ms: i64, // close time ms
    pub symbol_id: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone)]
struct WriterMsg {
    rows: Vec<CandleRow>,
}

// ---------- REST (zero-copy-ish) ----------
#[allow(dead_code)]
#[derive(serde::Deserialize)]
struct RestKline<'a>(
    i64,                       // open_time
    #[serde(borrow)] Cow<'a, str>, // open
    #[serde(borrow)] Cow<'a, str>, // high
    #[serde(borrow)] Cow<'a, str>, // low
    #[serde(borrow)] Cow<'a, str>, // close
    #[serde(borrow)] Cow<'a, str>, // volume
    i64,                       // close_time
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
);

// ---------- WS (zero-copy-ish) ----------
#[derive(serde::Deserialize)]
struct WsEnvelope<'a> {
    #[serde(borrow)]
    data: WsData<'a>,
}

#[derive(serde::Deserialize)]
struct WsData<'a> {
    #[serde(rename = "k", borrow)]
    k: WsKline<'a>,
}

#[derive(serde::Deserialize)]
struct WsKline<'a> {
    #[serde(rename = "s", borrow)]
    symbol: Cow<'a, str>, // "BTCUSDT"
    #[serde(rename = "i", borrow)]
    interval: Cow<'a, str>, // "1m"
    #[serde(rename = "T")]
    close_time: i64,
    #[serde(rename = "o", borrow)]
    open: Cow<'a, str>,
    #[serde(rename = "h", borrow)]
    high: Cow<'a, str>,
    #[serde(rename = "l", borrow)]
    low: Cow<'a, str>,
    #[serde(rename = "c", borrow)]
    close: Cow<'a, str>,
    #[serde(rename = "v", borrow)]
    volume: Cow<'a, str>,
    #[serde(rename = "x")]
    is_closed: bool,
}

fn parse_tf(s: &str) -> Option<TimeFrame> {
    match s {
        "1m" => Some(TimeFrame::M1),
        "5m" => Some(TimeFrame::M5),
        "15m" => Some(TimeFrame::M15),
        "1h" => Some(TimeFrame::H1),
        "4h" => Some(TimeFrame::H4),
        "1d" => Some(TimeFrame::D1),
        _ => None,
    }
}

fn build_table_name(tf: TimeFrame) -> String {
    format!("market.candles_{}", tf.as_str())
}

fn build_ws_base(ws_base: &str) -> String {
    let b = ws_base.trim_end_matches('/');
    if b.ends_with("/stream") {
        b.to_string()
    } else {
        format!("{}/stream", b)
    }
}

fn build_combined_ws_url(ws_base: &str, streams: &[String]) -> String {
    let base = build_ws_base(ws_base);
    let joined = streams.join("/");
    format!("{}?streams={}", base, joined)
}

fn str_f64(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

async fn pg_connect(db_url: &str) -> Result<tokio_postgres::Client> {
    let (client, connection) = tokio_postgres::connect(db_url, NoTls)
        .await
        .context("tokio_postgres::connect failed")?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            error!("postgres connection error: {e}");
        }
    });

    Ok(client)
}

async fn fetch_active_pairs(client: &tokio_postgres::Client) -> Result<Vec<PairInfo>> {
    let row = client
        .query_one("SELECT to_regclass('market.pairs')::text", &[])
        .await
        .context("failed to check market.pairs existence")?;
    let reg: Option<String> = row.get(0);
    if reg.is_none() {
        return Err(anyhow!(
            "DB schema is not initialized: relation market.pairs does not exist"
        ));
    }

    let rows = client
        .query(
            "SELECT symbol_id, symbol
             FROM market.pairs
             WHERE is_active = TRUE
             ORDER BY symbol_id",
            &[],
        )
        .await
        .context("fetch_active_pairs query failed")?;

    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        out.push(PairInfo {
            symbol_id: r.get::<_, i64>(0),
            symbol: r.get::<_, String>(1),
        });
    }
    Ok(out)
}

async fn fetch_last_time_per_symbol(
    client: &tokio_postgres::Client,
    tf: TimeFrame,
) -> Result<HashMap<i64, i64>> {
    let table = build_table_name(tf);
    let q = format!(
        "SELECT symbol_id, COALESCE(MAX(time_ms),0)::bigint AS last_ms
         FROM {}
         GROUP BY symbol_id",
        table
    );

    let rows = client
        .query(&q, &[])
        .await
        .with_context(|| format!("fetch_last_time_per_symbol failed for {}", table))?;

    let mut map = HashMap::with_capacity(rows.len().max(1024));
    for r in rows {
        let sid: i64 = r.get(0);
        let last_ms: i64 = r.get(1);
        map.insert(sid, last_ms);
    }
    Ok(map)
}

/// Подготовка буфера перед COPY:
/// - отбрасываем всё, что <= last_seen[symbol]
/// - сортируем и dedup по (symbol_id, time_ms)
fn prepare_copy_batch(buf: &mut Vec<CandleRow>, last_seen: &HashMap<i64, i64>) {
    buf.retain(|c| c.time_ms > *last_seen.get(&c.symbol_id).unwrap_or(&0));

    if buf.len() <= 1 {
        return;
    }

    buf.sort_unstable_by(|a, b| {
        match a.symbol_id.cmp(&b.symbol_id) {
            std::cmp::Ordering::Equal => a.time_ms.cmp(&b.time_ms),
            other => other,
        }
    });

    buf.dedup_by(|a, b| a.symbol_id == b.symbol_id && a.time_ms == b.time_ms);
}

/// BINARY COPY flush.
async fn flush_copy(
    client: &tokio_postgres::Client,
    tf: TimeFrame,
    buf: &mut Vec<CandleRow>,
    last_seen: &mut HashMap<i64, i64>,
) -> Result<u64> {
    if buf.is_empty() {
        return Ok(0);
    }

    prepare_copy_batch(buf, last_seen);
    if buf.is_empty() {
        return Ok(0);
    }

    let mut max_per_symbol: HashMap<i64, i64> = HashMap::new();
    for c in buf.iter() {
        let e = max_per_symbol.entry(c.symbol_id).or_insert(0);
        if c.time_ms > *e {
            *e = c.time_ms;
        }
    }

    let table_name = build_table_name(tf);
    let stmt = format!(
        "COPY {} (time_ms, time, symbol_id, open, high, low, close, volume) FROM STDIN BINARY",
        table_name
    );

    let sink = client.copy_in(&stmt).await.context("copy_in failed")?;
    let writer = BinaryCopyInWriter::new(
        sink,
        &[
            Type::INT8,   // time_ms
            Type::TIMESTAMPTZ, // time
            Type::INT8,   // symbol_id
            Type::FLOAT8, // open
            Type::FLOAT8, // high
            Type::FLOAT8, // low
            Type::FLOAT8, // close
            Type::FLOAT8, // volume
        ],
    );
    let mut writer = std::pin::pin!(writer);

    for r in buf.iter() {
        let ts = Utc.timestamp_millis_opt(r.time_ms).single().unwrap();
        writer
            .as_mut()
            .write(&[
                &r.time_ms,
                &ts,
                &r.symbol_id,
                &r.open,
                &r.high,
                &r.low,
                &r.close,
                &r.volume,
            ])
            .await?;
    }

    let rows = writer.as_mut().finish().await?;

    for (sid, mx) in max_per_symbol {
        let e = last_seen.entry(sid).or_insert(0);
        if mx > *e {
            *e = mx;
        }
    }

    buf.clear();
    Ok(rows)
}

async fn writer_task_copy(
    db_url: String,
    tf: TimeFrame,
    mut rx: mpsc::Receiver<WriterMsg>,
    mut last_seen: HashMap<i64, i64>,
    mut shutdown: watch::Receiver<bool>,
) -> Result<()> {
    let flush_rows: usize = std::env::var("INGEST_COPY_FLUSH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30_000);

    let flush_every_ms: u64 = std::env::var("INGEST_COPY_FLUSH_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(250);

    let mut inserted_total: u64 = 0;
    let mut last_log = Instant::now();

    loop {
        // Reconnect loop
        if *shutdown.borrow() {
            return Ok(());
        }

        let client = match pg_connect(&db_url).await {
            Ok(c) => c,
            Err(e) => {
                error!(?e, "pg_connect failed in writer task");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        };

        if std::env::var("INGEST_SYNC_COMMIT_OFF").ok().as_deref() == Some("1") {
            if let Err(e) = client.batch_execute("SET synchronous_commit = OFF;").await {
                error!(?e, "Failed to set synchronous_commit=OFF");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue; // reconnect
            }
            warn!("writer({}): synchronous_commit=OFF", tf.as_str());
        }

        let mut buf: Vec<CandleRow> = Vec::with_capacity(flush_rows * 2);
        let mut tick = tokio::time::interval(Duration::from_millis(flush_every_ms));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        let res: Result<()> = loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        break Ok(());
                    }
                }

                _ = tick.tick() => {
                    if !buf.is_empty() {
                        match flush_copy(&client, tf, &mut buf, &mut last_seen).await {
                            Ok(n) => {
                                inserted_total += n;
                                if last_log.elapsed() >= Duration::from_secs(2) {
                                    info!("writer({}): inserted {} rows, total {}", tf.as_str(), n, inserted_total);
                                    last_log = Instant::now();
                                }
                            }
                            Err(e) => {
                                error!(tf=?tf, error=format!("{:#}", e), "writer periodic flush failed");
                                break Err(e); // Break inner loop to reconnect
                            }
                        }
                    }
                }

                msg = rx.recv() => {
                    match msg {
                        Some(m) => {
                            buf.extend(m.rows);
                            if buf.len() >= flush_rows {
                                match flush_copy(&client, tf, &mut buf, &mut last_seen).await {
                                    Ok(n) => inserted_total += n,
                                    Err(e) => {
                                        error!(tf=?tf, error=format!("{:#}", e), "writer flush failed");
                                        break Err(e); // Break inner loop to reconnect
                                    }
                                }
                            }
                        }
                        None => {
                            // Channel closed, probably shutdown
                             break Ok(());
                        }
                    }
                }
            }
        };

        // Final flush before breaking outer loop or reconnecting
        if !buf.is_empty() {
             match flush_copy(&client, tf, &mut buf, &mut last_seen).await {
                Ok(n) => inserted_total += n,
                Err(e) => {
                    error!(tf=?tf, error=format!("{:#}", e), "writer final flush failed");
                }
            }
        }

        if res.is_ok() {
            // Channel closed, so exit.
            info!("writer({}) channel closed. inserted_total={}", tf.as_str(), inserted_total);
            return Ok(());
        }

        // Error happened, sleep before reconnect
        warn!("writer({}) reconnecting after error...", tf.as_str());
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}
async fn rest_fetch_klines_bytes(
    http: &reqwest::Client,
    cfg: &AppConfig,
    symbol: &str,
    interval: &str,
    start_time_ms: Option<i64>,
    limit: usize,
) -> Result<bytes::Bytes> {
    let mut url = format!(
        "{}/fapi/v1/klines?symbol={}&interval={}&limit={}",
        cfg.binance.rest_base_url, symbol, interval, limit
    );
    if let Some(st) = start_time_ms {
        url.push_str(&format!("&startTime={}", st));
    }

    let mut backoff = cfg.binance.http_retry_backoff_ms.max(50);
    for attempt in 1..=cfg.binance.http_retries.max(1) {
        let resp = http.get(&url).send().await;

        match resp {
            Ok(r) => {
                let status = r.status();
                if status.as_u16() == 429 || status.as_u16() == 418 {
                    warn!(
                        "REST rate-limited {} {} attempt={} status={}",
                        symbol, interval, attempt, status
                    );
                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                    backoff = (backoff * 2).min(cfg.binance.http_retry_backoff_max_ms);
                    continue;
                }

                match r.error_for_status() {
                    Ok(ok) => return Ok(ok.bytes().await.context("read bytes failed")?),
                    Err(e) => {
                        warn!(
                            "REST klines {} {} attempt={} status_err={}",
                            symbol, interval, attempt, e
                        );
                    }
                }
            }
            Err(e) => {
                warn!(
                    "REST klines {} {} attempt={} req_err={}",
                    symbol, interval, attempt, e
                );
            }
        }

        tokio::time::sleep(Duration::from_millis(backoff)).await;
        backoff = (backoff * 2).min(cfg.binance.http_retry_backoff_max_ms);
    }

    Err(anyhow!("REST klines failed after retries: {} {}", symbol, interval))
}

async fn rest_backfill_one(
    http: &reqwest::Client,
    cfg: &AppConfig,
    pair: &PairInfo,
    tf: TimeFrame,
    last_ms: i64,
    tx: mpsc::Sender<WriterMsg>,
) -> Result<()> {
    let interval = tf.as_str();
    let limit = cfg.runtime.backfill_candles.min(1000) as usize;

    let max_loops: usize = std::env::var("INGEST_BACKFILL_MAX_LOOPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);

    let mut start = if last_ms > 0 { Some(last_ms + 1) } else { None };
    let mut loops = 0usize;
    let mut max_close = last_ms;

    while loops < max_loops {
        loops += 1;

        let bytes = rest_fetch_klines_bytes(http, cfg, &pair.symbol, interval, start, limit).await?;
        let klines: Vec<RestKline> =
            serde_json::from_slice(&bytes).context("parse REST klines failed")?;
        if klines.is_empty() {
            break;
        }

        let mut out: Vec<CandleRow> = Vec::with_capacity(klines.len());

        for k in &klines {
            let close_time = k.6;
            if close_time <= last_ms {
                continue;
            }
            if close_time > max_close {
                max_close = close_time;
            }

            out.push(CandleRow {
                time_ms: close_time,
                symbol_id: pair.symbol_id,
                open: str_f64(k.1.as_ref()),
                high: str_f64(k.2.as_ref()),
                low: str_f64(k.3.as_ref()),
                close: str_f64(k.4.as_ref()),
                volume: str_f64(k.5.as_ref()),
            });
        }

        if !out.is_empty() {
            let _ = tx.send(WriterMsg { rows: out }).await;
        }

        if klines.len() < limit {
            break;
        }
        start = Some(max_close + 1);
    }

    Ok(())
}

async fn ws_worker(
    cfg: Arc<AppConfig>,
    url: String,
    symbol_to_id: Arc<HashMap<String, i64>>,
    writers: Arc<HashMap<TimeFrame, mpsc::Sender<WriterMsg>>>,
    mut shutdown: watch::Receiver<bool>,
) -> Result<()> {
    let mut backoff = cfg.binance.ws_reconnect_backoff_ms.max(200);
    let ping_every = Duration::from_secs(cfg.binance.ws_ping_interval_sec.max(5) as u64);

    loop {
        if *shutdown.borrow() {
            return Ok(());
        }

        info!("WS connect: {}", url);
        let conn = tokio_tungstenite::connect_async(&url).await;
        let (mut ws, _) = match conn {
            Ok(v) => v,
            Err(e) => {
                warn!("WS connect failed: {} err={}", url, e);
                tokio::time::sleep(Duration::from_millis(backoff)).await;
                backoff = (backoff * 2).min(cfg.binance.ws_reconnect_backoff_max_ms);
                continue;
            }
        };

        backoff = cfg.binance.ws_reconnect_backoff_ms.max(200);
        let mut ping = tokio::time::interval(ping_every);

        loop {
            tokio::select! {
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        let _ = ws.send(Message::Close(None)).await;
                        return Ok(());
                    }
                }

                _ = ping.tick() => {
                    // tungstenite 0.28: Ping(Bytes)
                    let _ = ws.send(Message::Ping(bytes::Bytes::new())).await;
                }

                msg = ws.next() => {
                    let msg = match msg {
                        None => { warn!("WS closed by server: {}", url); break; }
                        Some(Err(e)) => { warn!("WS read error: {} err={}", url, e); break; }
                        Some(Ok(m)) => m,
                    };

                    let data_slice: &[u8] = match &msg {
                        Message::Text(t) => t.as_bytes(),   // Utf8Bytes
                        Message::Binary(b) => b.as_ref(),   // Bytes
                        Message::Ping(_) | Message::Pong(_) => continue,
                        Message::Close(_) => { warn!("WS close frame: {}", url); break; }
                        _ => continue,
                    };

                    let env: WsEnvelope = match serde_json::from_slice(data_slice) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let k = env.data.k;
                    if !k.is_closed {
                        continue;
                    }

                    let tf = match parse_tf(k.interval.as_ref()) {
                        Some(t) => t,
                        None => continue,
                    };

                    let sid = match symbol_to_id.get(k.symbol.as_ref()) {
                        Some(v) => *v,
                        None => continue,
                    };

                    let row = CandleRow {
                        time_ms: k.close_time,
                        symbol_id: sid,
                        open: str_f64(k.open.as_ref()),
                        high: str_f64(k.high.as_ref()),
                        low: str_f64(k.low.as_ref()),
                        close: str_f64(k.close.as_ref()),
                        volume: str_f64(k.volume.as_ref()),
                    };

                    if let Some(tx) = writers.get(&tf) {
                        let _ = tx.send(WriterMsg { rows: vec![row] }).await;
                    }
                }
            }
        }

        tokio::time::sleep(Duration::from_millis(backoff)).await;
        backoff = (backoff * 2).min(cfg.binance.ws_reconnect_backoff_max_ms);
    }
}

/// Главная точка запуска свечей (REST backfill + WS realtime + BINARY COPY writers)
pub async fn run_candles_ingest() -> Result<()> {
    let cfg: AppConfig = load_config().context("load_config() failed")?;
    let cfg = Arc::new(cfg);

    let db_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| cfg.database.url());

    let http = reqwest::Client::builder()
        .timeout(Duration::from_millis(cfg.binance.http_timeout_ms.max(1_000)))
        .build()
        .context("reqwest build failed")?;

    let meta = pg_connect(&db_url).await?;
    let pairs = fetch_active_pairs(&meta).await?;
    if pairs.is_empty() {
        anyhow::bail!("No active pairs in market.pairs");
    }
    info!("Active pairs: {}", pairs.len());

    let mut map = HashMap::with_capacity(pairs.len() * 2);
    for p in &pairs {
        map.insert(p.symbol.clone(), p.symbol_id);
    }
    let symbol_to_id = Arc::new(map);

    let mut tfs: Vec<TimeFrame> = Vec::new();
    for s in &cfg.runtime.timeframes {
        if let Some(tf) = parse_tf(s) {
            tfs.push(tf);
        } else {
            warn!("Unknown timeframe in runtime.timeframes: {}", s);
        }
    }
    if tfs.is_empty() {
        anyhow::bail!("runtime.timeframes is empty or invalid");
    }

    let mut ws_tfs: Vec<TimeFrame> = Vec::new();
    for s in &cfg.runtime.realtime_ws_timeframes {
        if let Some(tf) = parse_tf(s) {
            ws_tfs.push(tf);
        } else {
            warn!("Unknown timeframe in runtime.realtime_ws_timeframes: {}", s);
        }
    }
    if ws_tfs.is_empty() {
        ws_tfs = tfs.clone();
    }

    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // writers (binary copy)
    let mut writers: HashMap<TimeFrame, mpsc::Sender<WriterMsg>> = HashMap::new();
    for tf in tfs.iter().copied() {
        let chan_size: usize = std::env::var("INGEST_WRITER_CHAN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50_000);

        let (tx, rx) = mpsc::channel::<WriterMsg>(chan_size);

        let last_seen = fetch_last_time_per_symbol(&meta, tf).await.unwrap_or_default();
        info!("last_seen loaded for {}: {}", tf.as_str(), last_seen.len());

        let db_url2 = db_url.clone();
        let shutdown_rx2 = shutdown_rx.clone();

        tokio::spawn(async move {
            if let Err(e) = writer_task_copy(db_url2, tf, rx, last_seen, shutdown_rx2).await {
                error!("writer({}) failed: {}", tf.as_str(), e);
            }
        });

        writers.insert(tf, tx);
    }
    let writers = Arc::new(writers);

    // WS combined streams, chunked
    let max_streams_per_ws: usize = std::env::var("INGEST_WS_MAX_STREAMS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);

    let mut streams: Vec<String> = Vec::with_capacity(pairs.len() * ws_tfs.len());
    for p in &pairs {
        let sym = p.symbol.to_lowercase();
        for tf in &ws_tfs {
            streams.push(format!("{}@kline_{}", sym, tf.as_str()));
        }
    }

    let ws_base = cfg.binance.ws_base_url.clone();
    let chunks: Vec<Vec<String>> = streams.chunks(max_streams_per_ws).map(|c| c.to_vec()).collect();
    info!("WS streams total={}, connections={}", streams.len(), chunks.len());

    for (i, chunk) in chunks.into_iter().enumerate() {
        let url = build_combined_ws_url(&ws_base, &chunk);
        let cfg2 = cfg.clone();
        let writers2 = writers.clone();
        let symbol_to_id2 = symbol_to_id.clone();
        let shutdown_rx2 = shutdown_rx.clone();

        tokio::spawn(async move {
            info!("WS worker #{} streams={}", i, chunk.len());
            if let Err(e) = ws_worker(cfg2, url, symbol_to_id2, writers2, shutdown_rx2).await {
                error!("WS worker #{} failed: {}", i, e);
            }
        });
    }

    // REST backfill concurrent
    let http_conc = cfg.runtime.http_concurrency.unwrap_or(1).max(1) as usize;
    info!(
        "REST backfill: tfs={}, pairs={}, http_concurrency={}",
        tfs.len(),
        pairs.len(),
        http_conc
    );

    let mut jobs = Vec::new();
    for tf in tfs.iter().copied() {
        let tx = writers.get(&tf).cloned().unwrap();
        let last = fetch_last_time_per_symbol(&meta, tf).await.unwrap_or_default();

        for p in &pairs {
            let last_ms = last.get(&p.symbol_id).copied().unwrap_or(0);
            let cfg2 = cfg.clone();
            let http2 = http.clone();
            let p2 = p.clone();
            let tx2 = tx.clone();

            jobs.push(async move {
                let r = rest_backfill_one(&http2, &cfg2, &p2, tf, last_ms, tx2).await;
                if let Err(e) = &r {
                    warn!("backfill {} {} failed: {}", p2.symbol, tf.as_str(), e);
                }
                r
            });
        }
    }

    let backfill_started = Instant::now();

    let (ok_cnt, err_cnt) = stream::iter(jobs)
        .buffer_unordered(http_conc)
        .fold((0u64, 0u64), |(ok, err), res| async move {
            match res {
                Ok(_) => (ok + 1, err),
                Err(_) => (ok, err + 1),
            }
        })
        .await;

    info!(
        "REST backfill done: ok={}, err={}, elapsed_ms={}",
        ok_cnt,
        err_cnt,
        backfill_started.elapsed().as_millis()
    );

    info!("Candles ingest running (WS realtime + BINARY COPY writers). Press Ctrl+C to stop.");
    tokio::signal::ctrl_c().await.context("ctrl_c failed")?;
    info!("Shutdown signal received.");

    let _ = shutdown_tx.send(true);
    tokio::time::sleep(Duration::from_millis(500)).await;

    Ok(())
}
