use anyhow::{Context, Result};
use std::{collections::HashMap, path::PathBuf, time::Duration};
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
                "svc_live".into(),
                "svc_writer".into(),
                "svc_compute".into(),
                "svc_orders".into(),
                "svc_position_tracker".into(),
                "svc_market_ingest".into(),
                "svc_webgui".into(),
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

    // 2) spawn daemons
    let mut children = Vec::new();
    for svc in services.iter() {
        let Some(spec) = specs.get(svc.as_str()) else {
            continue;
        };
        if spec.oneshot {
            continue;
        }

        info!("runner: spawn {}", spec.name);
        let mut cmd = Command::new(dir.join(spec.name));
        cmd.args(&spec.args);

        let child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn {}", spec.name))?;
        children.push((spec.name.to_string(), child));
    }

    info!("runner: all daemons spawned. Ctrl+C to stop.");
    tokio::select! {
        _ = signal::ctrl_c() => {
            warn!("runner: ctrl_c received, stopping...");
        }
        _ = async {
            loop {
                tokio::time::sleep(Duration::from_millis(500)).await;
                for (name, child) in children.iter_mut() {
                    if let Ok(Some(status)) = child.try_wait() {
                        error!("runner: child exited: {name} status={status}");
                        // Вместо bail! — просто логируем и выходим из select!
                        return; // ← выходим из async-блока, возвращая ()
                    }
                }
            }
        } => {}
    }

    // 3) graceful terminate
    for (name, mut child) in children {
        info!("runner: killing {name}...");
        let _ = child.kill().await.ok(); // ← вот так
    }

    info!("runner: stopped");
    Ok(())
}
