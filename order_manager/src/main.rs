// order_manager/src/main.rs
//
// Order Manager — точка входа.
//
// Запускает три параллельных async-задачи:
//   1. Signal Scanner loop  — сканирует БД на свежие сигналы
//   2. Position Tracker loop — отслеживает позиции, PnL, candles_left
//   3. Kafka Command listener — слушает orders.cmd от risk_manager
//
// При нахождении свободных слотов (< max_open_positions) — Scanner ищет
// сигналы и Executor открывает новые позиции. При достижении candles_left=0
// или SL/TP hit — Tracker инициирует закрытие через Executor.

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{info, warn, error};

use connections_lib::{BinanceFuturesClient, RedpandaConfig, RedpandaConnection};
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::Message;
use futures::StreamExt;

use order_manager::{
    OrderManagerConfig, SignalScanner, OrderExecutor, PositionTracker,
    PositionUpdateEvent, ExchangeInfoCache,
};
use risk_manager::position_closer::ClosePositionCommand;

#[tokio::main]
async fn main() -> Result<()> {
    // ─── Initialize ────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let _ = dotenvy::dotenv();

    info!("═══════════════════════════════════════════════════");
    info!("  🤖 Order Manager starting...");
    info!("═══════════════════════════════════════════════════");

    // Load config
    let config = OrderManagerConfig::load_with_env()?;
    let allocation = config.effective_timeframe_allocation();

    info!("Allocation: {:?}", allocation.slots);
    if config.is_pump_dump() {
        info!("Strategy: PUMP_DUMP (target={}%, max_hold={}, sl_frac={})",
            config.pump_dump.target_pct, config.pump_dump.max_hold_bars, config.pump_dump.sl_fraction);
        info!("PD pred min per TF: 5m={}, 15m={}, 1h={}, 4h={}, 1d={}",
            config.pump_dump.pd_pred_min_5m, config.pump_dump.pd_pred_min_15m,
            config.pump_dump.pd_pred_min_1h, config.pump_dump.pd_pred_min_4h,
            config.pump_dump.pd_pred_min_1d);
    } else {
        info!("P(super) min per TF: 1m={}, 5m={}, 15m={}, 1h={}, 4h={}, 1d={}",
            config.p_super_min_1m, config.p_super_min_5m, config.p_super_min_15m,
            config.p_super_min_1h, config.p_super_min_4h, config.p_super_min_1d);
    }
    info!("Max drift: {}%, Symbol cooldown: {}h, Max hold bars: {}",
        config.max_price_drift_pct, config.symbol_cooldown_hours,
        config.effective_max_hold_bars());

    // Load exchange credentials (retry-friendly — don't crash on failure)
    let exchange_settings = match settings_lib::ExchangeSettings::load_with_env() {
        Ok(es) => es,
        Err(e) => {
            warn!("Failed to load exchange settings: {}. Using defaults.", e);
            settings_lib::ExchangeSettings::default()
        }
    };

    if !exchange_settings.has_credentials() {
        error!("❌ No API credentials found!");
        error!("  Create ~/.settings.json with {{\"api_key\": \"...\", \"api_secret\": \"...\"}}");
        error!("  Or set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.");
        anyhow::bail!("No API credentials. Cannot trade without Binance keys.");
    }

    // All settings now in config/order_manager.toml (single source of truth)
    info!("Config: max_orders={}, leverage={}x, trade_size={} USDT ({}), strategy={}",
        config.max_orders_at_a_time, config.leverage, config.trade_size_value,
        config.trade_size_type, config.strategy_type);

    match config.trading_mode.as_str() {
        "off" => {
            warn!("⚠️ Trading mode is OFF — order_manager will run but NOT scan or trade.");
            warn!("  Click START TRADING in WebUI or set trading_mode = \"auto\" in order_manager.toml");
        }
        "manual" => {
            info!("📋 Trading mode: MANUAL — signals scanned, positions NOT opened automatically.");
        }
        "auto" => {
            warn!("🤖 Trading mode: AUTO — bot will automatically open and manage positions!");
        }
        other => {
            warn!("⚠️ Unknown trading_mode: '{}'. Using OFF.", other);
        }
    }

    // Create Binance Futures client
    let binance_client = BinanceFuturesClient::from_settings(&exchange_settings)?;

    // Test connectivity (don't crash on failure — just warn)
    let status = binance_client.check_connection().await;
    info!("Binance connection: {}", status);
    if !status.futures_auth {
        error!("⚠️ Binance Futures authentication failed: {:?}", status.error);
        error!("  order_manager will continue running and retry on each scan cycle.");
    }
    if !status.can_trade {
        error!("⚠️ Binance account cannot trade (can_trade=false).");
        error!("  Check that your API key has Futures trading permission.");
    }

    // Connect to database
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(10)
        .min_connections(2)
        .acquire_timeout(Duration::from_secs(10))
        .idle_timeout(Duration::from_secs(300))
        .connect(&config.database_url)
        .await?;

    info!("✅ Database connected");

    // Setup Redpanda producer for WebUI events
    let mut redpanda = RedpandaConnection::new(RedpandaConfig {
        brokers: config.kafka_brokers.clone(),
        topic: config.topic_positions.clone(),
        group_id: config.kafka_group_id.clone(),
    })?;
    redpanda.connect_producer().await?;
    info!("✅ Redpanda producer connected");

    let redpanda = Arc::new(redpanda);

    // Load Exchange Info cache (symbol precision, stepSize, etc.)
    // FIX: Don't crash if Binance API is unreachable at startup (e.g., VPN down).
    // Exchange info will be lazily refreshed on first trade attempt.
    let exchange_info = ExchangeInfoCache::new(binance_client.clone());
    match exchange_info.refresh().await {
        Ok(()) => {
            info!("✅ Exchange info cached: {} symbols", exchange_info.len().await);
        }
        Err(e) => {
            warn!("⚠️ Failed to load exchange info at startup: {}. Will retry on first trade.", e);
            warn!("  Check VPN/proxy if Binance API is blocked in your region.");
        }
    }

    // Create shared components
    let scanner = Arc::new(SignalScanner::new(pool.clone(), config.clone()));
    let executor = Arc::new(OrderExecutor::new(
        pool.clone(),
        config.clone(),
        binance_client.clone(),
        exchange_info.clone(),
    ));
    let tracker = Arc::new(Mutex::new(PositionTracker::new(
        pool.clone(),
        config.clone(),
        binance_client.clone(),
    )));

    // Load existing open positions from DB
    let open_positions = executor.load_open_positions().await?;
    {
        let mut tracker_lock = tracker.lock().await;
        tracker_lock.init(open_positions).await;
        info!("📊 Current state: {}", tracker_lock.summary());
    }

    // NOTE: trading_mode is now re-read from config on EVERY scanner cycle.
    // This allows WebUI to toggle auto-trading at runtime by modifying order_manager.toml.

    // ─── Spawn Tasks ────────────────────────────────────────

    // FIX: Shared cooldown map for symbols that fail to open (e.g., minNotional).
    // Prevents infinite retry loops like BTCUSDT failing every 30s.
    let failed_symbol_cooldown: Arc<Mutex<FailedSymbolCooldown>> =
        Arc::new(Mutex::new(HashMap::new()));

    // Task 1: Signal Scanner + Order Executor loop
    let scanner_handle = {
        let scanner = scanner.clone();
        let executor = executor.clone();
        let tracker = tracker.clone();
        let config = config.clone();
        let allocation = allocation.clone();
        let redpanda = redpanda.clone();
        let failed_cooldown = failed_symbol_cooldown.clone();

        tokio::spawn(async move {
            loop {
                // Re-read trading settings from order_manager.toml every cycle
                // This allows WebUI to toggle trading mode and update params at runtime
                let current_auto = OrderManagerConfig::load()
                    .map(|c| c.is_auto())
                    .unwrap_or(false);

                if let Err(e) = run_scanner_cycle(
                    &scanner,
                    &executor,
                    &tracker,
                    &config,
                    &allocation,
                    &redpanda,
                    current_auto,
                    &failed_cooldown,
                )
                .await
                {
                    error!("Scanner cycle error: {}", e);
                }

                tokio::time::sleep(Duration::from_secs(config.scan_interval_secs)).await;
            }
        })
    };

    // FIX #2: Shared retry counter for close failures
    let close_retries: Arc<Mutex<CloseRetryMap>> = Arc::new(Mutex::new(HashMap::new()));

    // Task 2: Position Tracker loop
    let tracker_handle = {
        let tracker = tracker.clone();
        let executor = executor.clone();
        let config = config.clone();
        let redpanda = redpanda.clone();
        let close_retries = close_retries.clone();

        tokio::spawn(async move {
            loop {
                if let Err(e) =
                    run_tracker_cycle(&tracker, &executor, &redpanda, &config, &close_retries).await
                {
                    error!("Tracker cycle error: {}", e);
                }

                tokio::time::sleep(Duration::from_secs(config.tracker_interval_secs)).await;
            }
        })
    };

    // Task 3: Kafka Command listener (orders.cmd from risk_manager)
    let cmd_handle = {
        let tracker = tracker.clone();
        let executor = executor.clone();
        let config = config.clone();
        let redpanda = redpanda.clone();

        tokio::spawn(async move {
            if let Err(e) =
                run_command_listener(&tracker, &executor, &redpanda, &config).await
            {
                error!("Command listener error: {}", e);
            }
        })
    };

    info!("═══════════════════════════════════════════════════");
    info!("  🚀 Order Manager running");
    info!("═══════════════════════════════════════════════════");

    // Wait for all tasks (they run forever)
    tokio::select! {
        r = scanner_handle => { error!("Scanner task exited: {:?}", r); }
        r = tracker_handle  => { error!("Tracker task exited: {:?}", r); }
        r = cmd_handle      => { error!("Command listener exited: {:?}", r); }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════
// SCANNER CYCLE
// ═══════════════════════════════════════════════════════════

/// Failed symbol cooldown: symbol → time of last failure.
/// Prevents infinite retry loops for symbols that consistently fail
/// (e.g., BTCUSDT minNotional errors every 30s scan cycle).
/// Cooldown: 10 minutes after failure before retrying.
type FailedSymbolCooldown = HashMap<String, Instant>;
const FAILED_SYMBOL_COOLDOWN_SECS: u64 = 600; // 10 minutes

async fn run_scanner_cycle(
    scanner: &SignalScanner,
    executor: &OrderExecutor,
    tracker: &Arc<Mutex<PositionTracker>>,
    config: &OrderManagerConfig,
    allocation: &order_manager::TimeframeAllocation,
    redpanda: &RedpandaConnection,
    is_auto: bool,
    failed_cooldown: &Arc<Mutex<FailedSymbolCooldown>>,
) -> Result<()> {
    // 1. Определить, какие слоты свободны
    let needed = scanner.needed_slots(allocation).await?;

    if needed.is_empty() {
        let tracker_lock = tracker.lock().await;
        tracing::debug!("All slots filled: {}", tracker_lock.summary());
        return Ok(());
    }

    info!("🔍 Scanner: need to fill slots: {:?}", needed);

    // 2. Сканировать сигналы
    let signals = scanner.scan_for_signals(&needed).await?;

    if signals.is_empty() {
        info!("📡 Scanner: no qualified signals found");
        return Ok(());
    }

    if !is_auto {
        info!(
            "📡 Scanner: found {} signals but mode=MANUAL, not opening",
            signals.len()
        );
        return Ok(());
    }

    // FIX #1: Track symbols opened during THIS cycle to prevent duplicate
    // orders for the same symbol on different timeframes within a single batch.
    // The DB check in scan_for_signals catches existing positions, but during
    // the same cycle, two signals (e.g., BREVUSDT 1h + BREVUSDT 4h) could slip through.
    let mut opened_symbols_this_cycle: HashSet<String> = HashSet::new();

    // Also pre-populate with symbols already in the tracker (in-memory, fresher than DB)
    {
        let tracker_lock = tracker.lock().await;
        for pos in tracker_lock.open_positions() {
            opened_symbols_this_cycle.insert(pos.symbol.clone());
        }
    }

    // Clean up expired failed-symbol cooldowns
    {
        let mut cooldown = failed_cooldown.lock().await;
        cooldown.retain(|_sym, failed_at| {
            failed_at.elapsed().as_secs() < FAILED_SYMBOL_COOLDOWN_SECS
        });
    }

    // 3. Открыть позиции для найденных сигналов
    for signal in &signals {
        // FIX #1: Skip if symbol already opened in this cycle or in tracker
        if opened_symbols_this_cycle.contains(&signal.symbol) {
            info!(
                "⚠️ Skipping {} tf={}m — symbol already has open position (anti-duplicate guard)",
                signal.symbol, signal.tf_minutes
            );
            continue;
        }

        // FIX: Skip if symbol recently failed to open (minNotional, etc.)
        // Prevents infinite retry loops like BTCUSDT failing every 30s
        {
            let cooldown = failed_cooldown.lock().await;
            if let Some(failed_at) = cooldown.get(&signal.symbol) {
                let elapsed = failed_at.elapsed().as_secs();
                if elapsed < FAILED_SYMBOL_COOLDOWN_SECS {
                    tracing::debug!(
                        "⏳ Skipping {} tf={}m — on failed cooldown ({}/{}s remaining)",
                        signal.symbol, signal.tf_minutes,
                        FAILED_SYMBOL_COOLDOWN_SECS - elapsed, FAILED_SYMBOL_COOLDOWN_SECS
                    );
                    continue;
                }
            }
        }

        // Проверяем, не заполнились ли слоты за время цикла
        let tracker_lock = tracker.lock().await;
        let current_tf_count = tracker_lock.count_by_tf(signal.tf_minutes);
        let target_slots = allocation.slots_for_tf(signal.tf_minutes);
        drop(tracker_lock);

        if current_tf_count as u16 >= target_slots {
            tracing::debug!(
                "Slot for tf={}m already filled ({}/{}), skipping",
                signal.tf_minutes, current_tf_count, target_slots
            );
            continue;
        }

        // Открыть позицию
        match executor.open_position(signal).await {
            Ok(managed_pos) => {
                // FIX #1: Mark symbol as opened to prevent duplicates in this batch
                opened_symbols_this_cycle.insert(managed_pos.symbol.clone());

                // Clear from failed cooldown on success
                {
                    let mut cooldown = failed_cooldown.lock().await;
                    cooldown.remove(&managed_pos.symbol);
                }

                // Отправить событие в WebUI
                let event = PositionUpdateEvent {
                    event_type: "position_opened".to_string(),
                    position_id: managed_pos.position_id,
                    symbol: managed_pos.symbol.clone(),
                    tf_minutes: managed_pos.tf_minutes,
                    side: managed_pos.side.as_str().to_string(),
                    entry_price: managed_pos.entry_price,
                    current_price: managed_pos.current_price,
                    sl_price: managed_pos.sl_price,
                    tp_price: managed_pos.tp_price,
                    qty: managed_pos.qty,
                    leverage: managed_pos.leverage,
                    unrealized_pnl: 0.0,
                    unrealized_pnl_pct: 0.0,
                    realized_pnl: None,
                    realized_pnl_pct: None,
                    candles_left: managed_pos.candles_left,
                    max_hold_bars: managed_pos.max_hold_bars,
                    combined_score: managed_pos.combined_score,
                    close_reason: None,
                    opened_at: managed_pos.opened_at.to_rfc3339(),
                    closed_at: None,
                    timestamp_ms: chrono::Utc::now().timestamp_millis(),
                };

                send_event(redpanda, config, &event).await;

                // Добавить в трекер
                let mut tracker_lock = tracker.lock().await;
                tracker_lock.add_position(managed_pos);
            }
            Err(e) => {
                // FIX: Add to failed cooldown to prevent retry loops
                {
                    let mut cooldown = failed_cooldown.lock().await;
                    cooldown.insert(signal.symbol.clone(), Instant::now());
                }
                warn!(
                    "❌ Failed to open position for {} tf={}m: {} (cooldown {}s)",
                    signal.symbol, signal.tf_minutes, e, FAILED_SYMBOL_COOLDOWN_SECS
                );
            }
        }
    }

    let tracker_lock = tracker.lock().await;
    info!("📊 After scanner: {}", tracker_lock.summary());

    Ok(())
}

// ═══════════════════════════════════════════════════════════
// TRACKER CYCLE
// ═══════════════════════════════════════════════════════════

/// FIX #2: Track close retry attempts to prevent infinite error loops.
/// After MAX_CLOSE_RETRIES, force-close the position in DB and remove from tracker.
const MAX_CLOSE_RETRIES: u32 = 5;

/// Shared retry counter for positions that failed to close on Binance.
/// Key = position_id, Value = retry count.
type CloseRetryMap = HashMap<i64, u32>;

async fn run_tracker_cycle(
    tracker: &Arc<Mutex<PositionTracker>>,
    executor: &OrderExecutor,
    redpanda: &RedpandaConnection,
    config: &OrderManagerConfig,
    close_retries: &Arc<Mutex<CloseRetryMap>>,
) -> Result<()> {
    let (events, to_close) = {
        let mut tracker_lock = tracker.lock().await;
        tracker_lock.tick().await?
    };

    // Отправить обновления в WebUI
    for event in &events {
        send_event(redpanda, config, event).await;
    }

    // Закрыть позиции, которые требуют закрытия
    for (position_id, reason) in &to_close {
        // FIX #2: Check if this position is already being retried too many times
        let retry_count = {
            let retries = close_retries.lock().await;
            retries.get(position_id).copied().unwrap_or(0)
        };

        if retry_count >= MAX_CLOSE_RETRIES {
            warn!(
                "🔴 Position #{} failed to close {} times. Force-closing in DB and removing from tracker.",
                position_id, retry_count
            );

            // FIX #2 + #7: Force-close in DB so it appears in history
            let pos = {
                let tracker_lock = tracker.lock().await;
                tracker_lock.get_position(*position_id).cloned()
            };

            if let Some(pos) = pos {
                // Try to mark as closed in DB with whatever info we have
                if let Err(e) = executor.force_close_in_db(&pos, *reason).await {
                    error!(
                        "❌ Failed to force-close position #{} in DB: {}",
                        position_id, e
                    );
                }
            }

            // Remove from tracker and retry map regardless
            {
                let mut tracker_lock = tracker.lock().await;
                tracker_lock.remove_position(*position_id);
            }
            {
                let mut retries = close_retries.lock().await;
                retries.remove(position_id);
            }
            continue;
        }

        let pos = {
            let tracker_lock = tracker.lock().await;
            tracker_lock.get_position(*position_id).cloned()
        };

        if let Some(pos) = pos {
            match executor.close_position(&pos, *reason, pos.current_price).await {
                Ok(close_event) => {
                    send_event(redpanda, config, &close_event).await;

                    let mut tracker_lock = tracker.lock().await;
                    tracker_lock.remove_position(*position_id);

                    // Clean up retry counter on success
                    let mut retries = close_retries.lock().await;
                    retries.remove(position_id);
                }
                Err(e) => {
                    // FIX #2: Increment retry counter instead of infinite loop
                    let mut retries = close_retries.lock().await;
                    let count = retries.entry(*position_id).or_insert(0);
                    *count += 1;
                    error!(
                        "❌ Failed to close position #{} (attempt {}/{}): {}",
                        position_id, count, MAX_CLOSE_RETRIES, e
                    );
                }
            }
        }
    }

    // Периодический лог
    if !events.is_empty() || !to_close.is_empty() {
        let tracker_lock = tracker.lock().await;
        info!("📊 Tracker: {}", tracker_lock.summary());
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════
// KAFKA COMMAND LISTENER (orders.cmd from risk_manager)
// ═══════════════════════════════════════════════════════════

async fn run_command_listener(
    tracker: &Arc<Mutex<PositionTracker>>,
    executor: &OrderExecutor,
    redpanda: &RedpandaConnection,
    config: &OrderManagerConfig,
) -> Result<()> {
    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", &config.kafka_brokers)
        .set("group.id", &config.kafka_group_id)
        .set("auto.offset.reset", "latest")
        .set("enable.auto.commit", "true")
        .create()?;

    consumer.subscribe(&[&config.topic_orders_cmd])?;
    info!("📥 Listening for commands on topic: {}", config.topic_orders_cmd);

    let mut stream = consumer.stream();

    while let Some(msg_result) = stream.next().await {
        match msg_result {
            Ok(msg) => {
                if let Some(payload) = msg.payload() {
                    // Try parsing as ClosePositionCommand first
                    if let Ok(cmd) = serde_json::from_slice::<ClosePositionCommand>(payload) {
                        if cmd.cmd_type == "close_all" {
                            // Emergency close all — from WebUI emergency_stop
                            warn!("🚨 Received CLOSE_ALL command from {}", cmd.source);

                            let all_positions: Vec<_> = {
                                let tracker_lock = tracker.lock().await;
                                tracker_lock.open_positions().to_vec()
                            };

                            for pos in &all_positions {
                                match executor
                                    .close_position(pos, order_manager::CloseReason::Manual, pos.current_price)
                                    .await
                                {
                                    Ok(close_event) => {
                                        send_event(redpanda, config, &close_event).await;
                                        let mut tracker_lock = tracker.lock().await;
                                        tracker_lock.remove_position(pos.position_id);
                                        warn!("🚨 Emergency closed position #{} {} {}", pos.position_id, pos.symbol, pos.side);
                                    }
                                    Err(e) => {
                                        error!("❌ Failed to emergency close position #{}: {}", pos.position_id, e);
                                    }
                                }
                            }
                        } else if cmd.cmd_type == "close_position" {
                            info!(
                                "📥 Received close command: {} {} from {}",
                                cmd.symbol, cmd.side, cmd.source
                            );

                            let to_close = {
                                let tracker_lock = tracker.lock().await;
                                tracker_lock.process_close_command(&cmd.symbol, &cmd.side)
                            };

                            for (position_id, reason) in &to_close {
                                let pos = {
                                    let tracker_lock = tracker.lock().await;
                                    tracker_lock.get_position(*position_id).cloned()
                                };

                                if let Some(pos) = pos {
                                    match executor
                                        .close_position(&pos, *reason, pos.current_price)
                                        .await
                                    {
                                        Ok(close_event) => {
                                            send_event(redpanda, config, &close_event).await;

                                            let mut tracker_lock = tracker.lock().await;
                                            tracker_lock.remove_position(*position_id);
                                        }
                                        Err(e) => {
                                            error!(
                                                "❌ Failed to close position #{} (risk_manager): {}",
                                                position_id, e
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Kafka consumer error: {}", e);
            }
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════

async fn send_event(
    redpanda: &RedpandaConnection,
    config: &OrderManagerConfig,
    event: &PositionUpdateEvent,
) {
    let key = format!("{}:{}", event.symbol, event.position_id);
    match serde_json::to_vec(event) {
        Ok(payload) => {
            if let Err(e) = redpanda
                .send_message(&config.topic_positions, &key, &payload)
                .await
            {
                warn!("Failed to send event to Redpanda: {}", e);
            }
        }
        Err(e) => {
            warn!("Failed to serialize event: {}", e);
        }
    }
}
