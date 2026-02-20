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
use std::sync::Arc;
use std::time::Duration;
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
    let allocation = config.timeframe_allocation();

    info!("Config: max_positions={}, leverage={}x, trade_size={} USDT",
        config.max_open_positions, config.leverage, config.trade_size_usdt);
    info!("Allocation: {:?}", allocation.slots);
    info!("Score range: [{}, {}], max_drift: {}%",
        config.signal_score_min, config.signal_score_max, config.max_price_drift_pct);

    // Load exchange credentials
    let exchange_settings = settings_lib::ExchangeSettings::load_with_env()?;
    if !exchange_settings.has_credentials() {
        anyhow::bail!(
            "No API credentials found. Set BINANCE_API_KEY/BINANCE_API_SECRET or create ~/.settings.json"
        );
    }

    // Check trading mode (off / manual / auto)
    let order_settings = settings_lib::OrderSettings::load_with_env()?;
    match order_settings.trading_mode {
        settings_lib::TradingMode::Off => {
            warn!("⚠️ Trading mode is OFF — order_manager will run but NOT scan or trade.");
            warn!("  Set ORDER_TRADING_MODE=manual or auto to enable.");
        }
        settings_lib::TradingMode::Manual => {
            info!("📋 Trading mode: MANUAL — signals will be scanned but positions will NOT open automatically.");
            info!("  Confirm orders via WebUI. Set ORDER_TRADING_MODE=auto for full automation.");
        }
        settings_lib::TradingMode::Auto => {
            warn!("🤖 Trading mode: AUTO — bot will automatically open and manage positions!");
        }
    }

    // Create Binance Futures client
    let binance_client = BinanceFuturesClient::from_settings(&exchange_settings)?;

    // Test connectivity
    let status = binance_client.check_connection().await;
    info!("Binance connection: {}", status);
    if !status.futures_auth {
        anyhow::bail!("Binance Futures authentication failed: {:?}", status.error);
    }
    if !status.can_trade {
        anyhow::bail!("Binance account cannot trade (can_trade=false)");
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
    let exchange_info = ExchangeInfoCache::new(binance_client.clone());
    exchange_info.refresh().await?;
    info!("✅ Exchange info cached: {} symbols", exchange_info.len().await);

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

    let is_auto = order_settings.trading_mode.is_auto();
    let _is_active = order_settings.trading_mode.is_active(); // true for auto/manual, false for off

    // ─── Spawn Tasks ────────────────────────────────────────

    // Task 1: Signal Scanner + Order Executor loop
    let scanner_handle = {
        let scanner = scanner.clone();
        let executor = executor.clone();
        let tracker = tracker.clone();
        let config = config.clone();
        let allocation = allocation.clone();
        let redpanda = redpanda.clone();

        tokio::spawn(async move {
            loop {
                if let Err(e) = run_scanner_cycle(
                    &scanner,
                    &executor,
                    &tracker,
                    &config,
                    &allocation,
                    &redpanda,
                    is_auto,
                )
                .await
                {
                    error!("Scanner cycle error: {}", e);
                }

                tokio::time::sleep(Duration::from_secs(config.scan_interval_secs)).await;
            }
        })
    };

    // Task 2: Position Tracker loop
    let tracker_handle = {
        let tracker = tracker.clone();
        let executor = executor.clone();
        let config = config.clone();
        let redpanda = redpanda.clone();

        tokio::spawn(async move {
            loop {
                if let Err(e) =
                    run_tracker_cycle(&tracker, &executor, &redpanda, &config).await
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

async fn run_scanner_cycle(
    scanner: &SignalScanner,
    executor: &OrderExecutor,
    tracker: &Arc<Mutex<PositionTracker>>,
    config: &OrderManagerConfig,
    allocation: &order_manager::TimeframeAllocation,
    redpanda: &RedpandaConnection,
    is_auto: bool,
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

    // 3. Открыть позиции для найденных сигналов
    for signal in &signals {
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
                error!(
                    "❌ Failed to open position for {} tf={}m: {}",
                    signal.symbol, signal.tf_minutes, e
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

async fn run_tracker_cycle(
    tracker: &Arc<Mutex<PositionTracker>>,
    executor: &OrderExecutor,
    redpanda: &RedpandaConnection,
    config: &OrderManagerConfig,
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
                }
                Err(e) => {
                    error!(
                        "❌ Failed to close position #{}: {}",
                        position_id, e
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
                    if let Ok(cmd) = serde_json::from_slice::<ClosePositionCommand>(payload) {
                        if cmd.cmd_type == "close_position" {
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
