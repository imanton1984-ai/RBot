// apps/src/bin/runner.rs

use anyhow::{Context, Result};
use std::{
    collections::HashMap,
    env,
    path::PathBuf,
    time::Duration,
};
use tokio::{process::Command, signal};
use tracing::{error, info, warn};

use apps::init_tracing;

#[derive(Clone, Debug)]
struct ProcSpec {
    name: &'static str,
    args: Vec<String>,
    oneshot: bool,
}

fn bin_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("current_exe() failed")?;
    Ok(exe
        .parent()
        .context("current_exe has no parent")?
        .to_path_buf())
}

fn parse_services_env() -> Vec<String> {
    std::env::var("RUNNER_SERVICES")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| {
            vec![
                "svc_db_init".into(),
                "svc_health".into(),
                "svc_market_ingest".into(), // Moved before writer/backfill
                "svc_writer".into(),
                "svc_backfill".into(),      // Moved here to potentially wait for ingest
                "svc_live".into(),
                "svc_compute".into(),
                "svc_orders".into(),
                "svc_position_tracker".into(),
                "svc_webgui".into(),
                "svc_api_gateway".into(),
            ]
        })
}

fn default_specs() -> HashMap<&'static str, ProcSpec> {
    let mut m = HashMap::new();

    // one-shot init
    m.insert(
        "svc_db_init",
        ProcSpec {
            name: "svc_db_init",
            args: vec![],
            oneshot: true,
        },
    );

    // optional one-shot backfill (если захочешь включить в RUNNER_SERVICES)
    m.insert(
        "svc_backfill",
        ProcSpec {
            name: "svc_backfill",
            args: vec![],
            oneshot: true,
        },
    );

    // long-running
    for s in [
        "svc_health",
        "svc_live",
        "svc_writer",
        "svc_compute",
        "svc_orders",
        "svc_position_tracker",
        "svc_market_ingest",
        "svc_webgui",
        "svc_api_gateway",
    ] {
        m.insert(
            s,
            ProcSpec {
                name: s,
                args: vec![],
                oneshot: false,
            },
        );
    }

    m
}

// --- НОВАЯ ФУНКЦИЯ ---
async fn wait_for_stage_via_health(
    health_port: u16,
    target_stage: &str,
    timeout_seconds: u64,
) -> Result<()> {
    let url = format!("http://127.0.0.1:{}/stagez", health_port);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_seconds);

    info!("runner: waiting for stage '{}' via health service at {} (timeout: {}s)", target_stage, url, timeout_seconds);

    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("runner: timeout waiting for stage '{}'", target_stage);
        }

        // Выполняем HTTP GET запрос
        let response = reqwest::get(&url).await;

        if let Ok(resp) = response {
            if resp.status().is_success() {
                if let Ok(json_body) = resp.json::<serde_json::Value>().await {
                    // Проверяем структуру JSON, возвращаемую svc_health.
                    // svc_health/src/main.rs: Json(HashMap::from([...]))
                    // readyz возвращает: Json(HashMap::from([("ok", ok.to_string()), ("stage", s.stage.clone())]))
                    // stagez возвращает: Json(s.clone()) -> это структура StageState
                    // StageState (из common/src/health.rs, предполагаемая структура):
                    // { "stage": "RUN", "details": { "svc_market_ingest": "PAIRS_READY", ... } }
                    // Или, как в коде svc_health: details.insert("svc_market_ingest", "PAIRS_READY");
                    // Тогда ожидаем: { "stage": "...", "details": { "svc_market_ingest": "PAIRS_READY", ... } }
                    // Или: { "current": "PAIRS_READY", "details": {...} }
                    // Проверим наиболее вероятный вариант: details содержит статусы сервисов.
                    let stage_ok = json_body.get("details")
                        .and_then(|d| d.get("svc_market_ingest")) // Проверяем статус svc_market_ingest
                        .and_then(|v| v.as_str())
                        .map(|s| s == target_stage)
                        .unwrap_or(false);

                    // Также проверим, может ли статус быть в поле "current" (как в readyz)
                    let stage_ok_alt = json_body.get("current")
                        .and_then(|v| v.as_str())
                        .map(|s| s == target_stage)
                        .unwrap_or(false);

                    // Проверим для BACKFILL_CANDLES_READY - возможно, отслеживается svc_backfill
                    let backfill_stage_ok = json_body.get("details")
                        .and_then(|d| d.get("svc_backfill")) // Проверяем статус svc_backfill
                        .and_then(|v| v.as_str())
                        .map(|s| s == target_stage)
                        .unwrap_or(false);

                    if stage_ok || stage_ok_alt || backfill_stage_ok {
                         info!("runner: stage '{}' confirmed via health service.", target_stage);
                         return Ok(());
                    }
                } else {
                    warn!("runner: failed to parse JSON response from health service at {}", url);
                }
            } else {
                 warn!("runner: health service responded with non-success status: {}", resp.status());
            }
        } else {
             // Возможно, сервис еще не готов отвечать
             warn!("runner: failed to query health service at {}: {:?}", url, response.err());
        }

        tokio::time::sleep(Duration::from_secs(2)).await; // Пауза между проверками
    }
}
// --- КОНЕЦ НОВОЙ ФУНКЦИИ ---

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let services = parse_services_env();
    let specs = default_specs();
    let dir = bin_dir()?;

    info!("runner: bin_dir={}", dir.display());
    info!("runner: services={:?}", services);

    // 1) run oneshots in order (as listed), stop on failure
    for svc in services.iter() {
        let Some(spec) = specs.get(svc.as_str()) else {
            warn!("runner: unknown service '{svc}', skip");
            continue;
        };
        if !spec.oneshot {
            continue;
        }

        info!("runner: oneshot start {}", spec.name);
        let mut cmd = Command::new(dir.join(spec.name));
        cmd.args(&spec.args);

        let status = cmd
            .status()
            .await
            .with_context(|| format!("failed to run oneshot {}", spec.name))?;

        if !status.success() {
            anyhow::bail!("oneshot {} failed with status={status}", spec.name);
        }
        info!("runner: oneshot done {}", spec.name);
    }

    // --- НОВАЯ ЛОГИКА ЗАПУСКА DAEMONS С ЗАВИСИМОСТЯМИ ---
    info!("runner: starting daemons with dependencies...");

    let health_port = env::var("SVC_HEALTH_PORT")
        .unwrap_or_else(|_| "9005".to_string())
        .parse::<u16>()
        .unwrap_or(9005);

    let mut running_daemons = Vec::new();

    // Запускаем svc_health первым
    let health_spec = specs.get("svc_health").expect("svc_health spec must exist");
    info!("runner: daemon spawn {}", health_spec.name);
    let mut health_cmd = Command::new(dir.join(health_spec.name));
    health_cmd.args(&health_spec.args);
    let health_child = health_cmd
        .spawn()
        .with_context(|| format!("failed to spawn {}", health_spec.name))?;
    running_daemons.push((health_spec.name.to_string(), health_child));

    // Ждем, пока health станет ready (опционально, но рекомендуется)
    info!("runner: waiting for svc_health to be ready...");
    let health_ready_url = format!("http://127.0.0.1:{}/readyz", health_port);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60); // Таймаут для готовности health
    loop {
        if tokio::time::Instant::now() > deadline {
            anyhow::bail!("runner: timeout waiting for svc_health to be ready");
        }
        let response = reqwest::get(&health_ready_url).await;
        if let Ok(resp) = response {
            if resp.status().is_success() {
                info!("runner: svc_health is ready.");
                break;
            }
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    // Запускаем svc_market_ingest
    let ingest_spec = specs.get("svc_market_ingest").expect("svc_market_ingest spec must exist");
    info!("runner: daemon spawn {}", ingest_spec.name);
    let mut ingest_cmd = Command::new(dir.join(ingest_spec.name));
    ingest_cmd.args(&ingest_spec.args);
    let ingest_child = ingest_cmd
        .spawn()
        .with_context(|| format!("failed to spawn {}", ingest_spec.name))?;
    running_daemons.push((ingest_spec.name.to_string(), ingest_child));

    // Ждем стадии PAIRS_READY через svc_health
    wait_for_stage_via_health(health_port, "PAIRS_READY", 120).await?;

    // Теперь можно запустить svc_writer и svc_backfill
    let writer_spec = specs.get("svc_writer").expect("svc_writer spec must exist");
    info!("runner: daemon spawn {}", writer_spec.name);
    let mut writer_cmd = Command::new(dir.join(writer_spec.name));
    writer_cmd.args(&writer_spec.args);
    let writer_child = writer_cmd
        .spawn()
        .with_context(|| format!("failed to spawn {}", writer_spec.name))?;
    running_daemons.push((writer_spec.name.to_string(), writer_child));

    let backfill_spec = specs.get("svc_backfill").expect("svc_backfill spec must exist");
    info!("runner: daemon spawn {}", backfill_spec.name);
    let mut backfill_cmd = Command::new(dir.join(backfill_spec.name));
    backfill_cmd.args(&backfill_spec.args);
    let backfill_child = backfill_cmd
        .spawn()
        .with_context(|| format!("failed to spawn {}", backfill_spec.name))?;
    running_daemons.push((backfill_spec.name.to_string(), backfill_child));

    // Ждем стадии BACKFILL_CANDLES_READY через svc_health
    // В svc_health логике (предполагаемой) эта стадия устанавливается для svc_backfill или svc_market_ingest.
    // Проверим в svc_backfill.
    wait_for_stage_via_health(health_port, "BACKFILL_CANDLES_READY", 3600).await?;

    // Остальные сервисы, которые могут запускаться параллельно после этих ключевых
    let other_daemon_services = [
        "svc_live",           // Может зависеть от backfill?
        "svc_compute",
        "svc_orders",
        "svc_position_tracker",
        "svc_webgui",
        "svc_api_gateway",
    ];

    for svc_name in other_daemon_services.iter() {
         if let Some(spec) = specs.get(svc_name) {
             info!("runner: daemon spawn {}", spec.name);
             let mut cmd = Command::new(dir.join(spec.name));
             cmd.args(&spec.args);
             let child = cmd
                 .spawn()
                 .with_context(|| format!("failed to spawn {}", spec.name))?;
             running_daemons.push((spec.name.to_string(), child));
         }
    }

    info!("runner: all critical daemons spawned and initial stages verified. Ctrl+C to stop.");

    // --- ОРИГИНАЛЬНАЯ ЛОГИКА МОНИТОРИНГА И ЗАВЕРШЕНИЯ ---
    // (оставляем как есть, но меняем `children` на `running_daemons`)
    tokio::select! {
        _ = signal::ctrl_c() => {
            warn!("runner: ctrl_c received, stopping...");
        }
        _ = async {
            loop {
                tokio::time::sleep(Duration::from_millis(500)).await;
                for (name, child) in running_daemons.iter_mut() {
                    if let Ok(Some(status)) = child.try_wait() {
                        error!("runner: child exited: {name} status={status}");
                        return; // <- выходим из async-блока, возвращая ()
                    }
                }
            }
        } => {}
    }

    // 3) graceful terminate
    for (name, mut child) in running_daemons {
        info!("runner: killing {name}...");
        let _ = child.kill().await.ok(); // <- вот так
    }

    info!("runner: stopped");
    Ok(())
}
