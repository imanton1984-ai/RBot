use anyhow::{Context, Result};
use std::{path::PathBuf, time::Duration};
use tokio_postgres::NoTls;
use tracing::{info, warn};
use walkdir::WalkDir;

use apps::init_tracing;

fn arg_value(flag: &str) -> Option<String> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == flag {
            return it.next();
        }
    }
    None
}

fn ddl_dir() -> PathBuf {
    // по умолчанию как у тебя в скриптах: "--dir database/ddl"
    arg_value("--dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("database/ddl"))
}

fn database_url() -> Result<String> {
    if let Ok(v) = std::env::var("DATABASE_URL") {
        return Ok(v);
    }
    // fallback на общий конфиг
    let cfg = common::config::load_config().context("load_config failed")?;
    Ok(cfg.database.url())
}

fn collect_sql_files(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for e in WalkDir::new(dir).into_iter().filter_map(|x| x.ok()) {
        if !e.file_type().is_file() {
            continue;
        }
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) == Some("sql") {
            files.push(p.to_path_buf());
        }
    }
    files.sort();
    Ok(files)
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let dir = ddl_dir();
    let db_url = database_url()?;

    info!("svc_db_init: using DATABASE_URL={}", db_url);
    info!("svc_db_init: ddl dir={}", dir.display());

    let sql_files = collect_sql_files(&dir)
        .with_context(|| format!("collect_sql_files failed for {}", dir.display()))?;
    if sql_files.is_empty() {
        warn!("svc_db_init: no .sql files found in {}", dir.display());
        return Ok(());
    }

    // connect
    let (client, conn) = tokio_postgres::connect(&db_url, NoTls)
        .await
        .context("tokio_postgres::connect failed")?;
    tokio::spawn(async move {
        if let Err(e) = conn.await {
            warn!("svc_db_init: db connection error: {e:#}");
        }
    });

    // apply sequentially (простая/надежная модель)
    for p in sql_files {
        let sql = tokio::fs::read_to_string(&p)
            .await
            .with_context(|| format!("read_to_string failed: {}", p.display()))?;

        let name = p.file_name().and_then(|x| x.to_str()).unwrap_or("<sql>");
        info!("svc_db_init: applying {name} ...");

        // batch_execute удобен для файлов с несколькими statements
        client
            .batch_execute(&sql)
            .await
            .with_context(|| format!("batch_execute failed: {}", p.display()))?;

        // микропаузa чтобы лог был “читаемым”, и не штурмовать DB на старте
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    info!("svc_db_init: done OK");
    Ok(())
}
