use std::{net::SocketAddr, sync::Arc, time::Duration};

use anyhow::Result;
use axum::{routing::get, Router};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch, RwLock};
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

use connections_lib::{
    binance_api::BinanceApi,
    binance_websocket::BinanceWsConnection,
    database::DatabaseConnection,
    redpanda::{RedpandaConfig, RedpandaConnection},
};

#[derive(Clone, Debug, serde::Serialize)]
struct HealthState {
    db_ok: bool,
    rp_ok: bool,
    api_ok: bool,
    ws_connected: bool,
    last_check_ms: u128,
    last_error: Option<String>,
}

impl Default for HealthState {
    fn default() -> Self {
        Self {
            db_ok: false,
            rp_ok: false,
            api_ok: false,
            ws_connected: false,
            last_check_ms: 0,
            last_error: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DepStatus {
    db_ok: bool,
    rp_ok: bool,
    api_ok: bool,
    ws_connected: bool,
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn log_status_transition(prev: DepStatus, cur: DepStatus, last_error: &Option<String>) {
    // Логируем только изменения.
    // "Падение" -> WARN, "восстановление" -> INFO.
    if prev.db_ok != cur.db_ok {
        if cur.db_ok {
            info!("db_ok -> true");
        } else {
            warn!("db_ok -> false");
        }
    }
    if prev.rp_ok != cur.rp_ok {
        if cur.rp_ok {
            info!("rp_ok -> true");
        } else {
            warn!("rp_ok -> false");
        }
    }
    if prev.api_ok != cur.api_ok {
        if cur.api_ok {
            info!("api_ok -> true");
        } else {
            warn!("api_ok -> false");
        }
    }
    if prev.ws_connected != cur.ws_connected {
        if cur.ws_connected {
            info!("ws_connected -> true");
        } else {
            warn!("ws_connected -> false");
        }
    }

    // Если статус изменился и есть ошибка — покажем её одной строкой.
    if let Some(e) = last_error {
        warn!("last_error: {e}");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    // Поддержка RUST_LOG (пример: RUST_LOG=info,connections_lib=debug)
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("Failed to install crypto provider");

    let health_addr: SocketAddr = env_string("CONNECTIONS_HEALTH_ADDR", "0.0.0.0:18080").parse()?;

    let state = Arc::new(RwLock::new(HealthState::default()));
    let (ws_tx, mut ws_rx) = mpsc::channel::<bytes::Bytes>(2000);
    let (status_tx, status_rx) = watch::channel(false);

    // Kafka Forwarder
    let mut rp = RedpandaConnection::new(RedpandaConfig::default())?;
    rp.connect_producer().await?;
    let rp_shared = Arc::new(rp);

    // топик оставляем прежним, но даём возможность переопределить из env (по умолчанию "ws_feed")
    let ws_topic = env_string("TOPIC_MARKET_WS_RAW", "ws_feed");

    let rp_for_ws = rp_shared.clone();
    tokio::spawn(async move {
        info!("Starting Kafka forwarder... topic={}", ws_topic);

        // тихая статистика: 1-е сообщение + дальше раз в N сообщений
        let mut n: u64 = 0;
        let mut last_log = tokio::time::Instant::now();

        while let Some(msg) = ws_rx.recv().await {
            n += 1;

            // Логируем только 1-е сообщение и дальше примерно раз в 10 секунд
            // (так видно, что поток жив, но без спама)
            if n == 1 || last_log.elapsed() >= Duration::from_secs(10) {
                info!("WS->Kafka flow ok: msgs={}, last_size={} bytes", n, msg.len());
                last_log = tokio::time::Instant::now();
            }

            // Отправляем байты в Kafka
            if let Err(e) = rp_for_ws.send_message(&ws_topic, "", msg).await {
                // Ошибка отправки — это важно, но тоже без спама: WARN/ERROR
                // Если будет сыпаться — ты сразу увидишь.
                error!("Kafka forward error: {e}");
            }
        }

        warn!("Kafka forwarder stopped (ws_rx closed)");
    });


    // Binance WS Task
    let ws_conn = BinanceWsConnection::new_from_env().await?;
    tokio::spawn(async move {
        if let Err(e) = ws_conn.run_forever(ws_tx, status_tx).await {
            error!("WS Task died: {e}");
        }
    });

    // Health Check Task: тихо, информативно, без пересоздания клиентов каждый тик
    let health_state = state.clone();
    let rp_health = rp_shared.clone();
    tokio::spawn(async move {
        let check_sec = env_u64("CONNECTIONS_CHECK_SEC", 10);
        let mut interval = tokio::time::interval(Duration::from_secs(check_sec));

        // Попытка создать клиентов один раз; при фейле — ретраим.
        let mut db: Option<DatabaseConnection> = None;
        let mut api: Option<BinanceApi> = None;

        let mut prev = DepStatus::default();

        loop {
            interval.tick().await;

            // ws_connected берём всегда (даже если db/api/rp не готовы)
            let ws_connected = *status_rx.borrow();

            // Ленивая инициализация DB
            if db.is_none() {
                match DatabaseConnection::new(Default::default()).await {
                    Ok(conn) => {
                        db = Some(conn);
                        debug!("DB client initialized");
                    }
                    Err(e) => {
                        // Если даже создать не можем — это уже существенная ошибка
                        let mut st = health_state.write().await;
                        st.db_ok = false;
                        st.ws_connected = ws_connected;
                        st.last_check_ms = now_ms();
                        st.last_error = Some(format!("DB init: {e}"));

                        let cur = DepStatus {
                            db_ok: false,
                            rp_ok: st.rp_ok, // rp_ok обновим ниже
                            api_ok: st.api_ok,
                            ws_connected,
                        };
                        if cur != prev {
                            log_status_transition(prev, cur, &st.last_error);
                            prev = cur;
                        }
                        continue;
                    }
                }
            }

            // Ленивая инициализация API
            if api.is_none() {
                match BinanceApi::new() {
                    Ok(client) => {
                        api = Some(client);
                        debug!("Binance API client initialized");
                    }
                    Err(e) => {
                        let mut st = health_state.write().await;
                        st.api_ok = false;
                        st.ws_connected = ws_connected;
                        st.last_check_ms = now_ms();
                        st.last_error = Some(format!("API init: {e}"));

                        let cur = DepStatus {
                            db_ok: st.db_ok,
                            rp_ok: st.rp_ok,
                            api_ok: false,
                            ws_connected,
                        };
                        if cur != prev {
                            log_status_transition(prev, cur, &st.last_error);
                            prev = cur;
                        }
                        continue;
                    }
                }
            }

            // Параллельные проверки
            let db_ref = db.as_ref().unwrap();
            let api_ref = api.as_ref().unwrap();

            let (db_res, api_res, rp_res) = tokio::join!(db_ref.ping(), api_ref.ping(), rp_health.ping());

            let db_ok = db_res.is_ok();
            let api_ok = api_res.is_ok();
            let rp_ok = rp_res.is_ok();

            // если ping упал — можно сбросить клиент (чтобы пересоздался на следующем тике)
            // это помогает при "сломанных" внутренних состояниях пула/клиента
            if db_res.is_err() {
                db = None;
            }
            if api_res.is_err() {
                api = None;
            }

            // Обновляем state
            let mut st = health_state.write().await;
            st.db_ok = db_ok;
            st.api_ok = api_ok;
            st.rp_ok = rp_ok;
            st.ws_connected = ws_connected;
            st.last_check_ms = now_ms();

            st.last_error = if let Err(e) = &db_res {
                Some(format!("DB: {e}"))
            } else if let Err(e) = &api_res {
                Some(format!("API: {e}"))
            } else if let Err(e) = &rp_res {
                Some(format!("RP: {e}"))
            } else {
                None
            };

            // Логируем только если поменялось
            let cur = DepStatus {
                db_ok,
                rp_ok,
                api_ok,
                ws_connected,
            };
            if cur != prev {
                log_status_transition(prev, cur, &st.last_error);
                prev = cur;
            } else {
                debug!("Health unchanged");
            }
        }
    });

    let app = Router::new().route(
        "/health",
        get(move || {
            let s = state.clone();
            async move {
                let st = s.read().await;
                axum::Json(st.clone())
            }
        }),
    );

    info!("Server starting on {}", health_addr);
    let listener = TcpListener::bind(health_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async { tokio::signal::ctrl_c().await.ok(); };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .unwrap()
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
}




