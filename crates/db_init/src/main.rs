use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use tokio_postgres::{NoTls};

fn default_sql_dir() -> PathBuf {
    PathBuf::from("infra/database/init")
}

fn is_sql_file(path: &Path) -> bool {
    path.extension().map(|e| e == "sql").unwrap_or(false)
}

fn list_sql_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = WalkDir::new(dir)
        .min_depth(1)
        .max_depth(2)
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_path_buf())
        .filter(|p| is_sql_file(p))
        .collect();

    files.sort(); // порядок по имени: 001_, 010_, ...
    Ok(files)
}

/// DATABASE_URL priority:
/// 1) env DATABASE_URL
/// 2) config/database.toml -> common::config::load_config()
fn resolve_db_url() -> Result<String> {
    if let Ok(url) = std::env::var("DATABASE_URL") {
        if !url.trim().is_empty() {
            return Ok(url);
        }
    }

    let cfg = common::config::load_config().context("failed to load config files")?;
    Ok(cfg.database.url())
}

#[tokio::main]
async fn main() -> Result<()> {
    // CLI: allow override SQL dir
    // Usage:
    //   cargo run -p db_init -- --dir infra/database/init
    let mut args = std::env::args().skip(1);
    let mut dir = default_sql_dir();

    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => {
                let v = args.next().context("--dir requires a value")?;
                dir = PathBuf::from(v);
            }
            "--help" | "-h" => {
                eprintln!("Usage: db_init [--dir <path>]");
                return Ok(());
            }
            other => {
                eprintln!("Unknown arg: {other}");
                eprintln!("Usage: db_init [--dir <path>]");
                std::process::exit(2);
            }
        }
    }

    let db_url = resolve_db_url().context("DATABASE_URL not set and config/database.toml missing")?;
    let sql_files = list_sql_files(&dir).with_context(|| format!("failed to list sql files in {}", dir.display()))?;

    if sql_files.is_empty() {
        anyhow::bail!("no .sql files found in {}", dir.display());
    }

    println!("db_init: connecting to DB...");
    let (client, connection) = tokio_postgres::connect(&db_url, NoTls)
        .await
        .context("failed to connect to postgres")?;

    // драйвер соединения
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("postgres connection error: {e}");
        }
    });

    println!("db_init: applying migrations from {}", dir.display());
    for path in sql_files {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("<unknown>");
        let sql = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;

        // Пропускаем пустые файлы/комментарии
        if sql.trim().is_empty() {
            println!(" - {name}: skipped (empty)");
            continue;
        }

        println!(" - {name}: executing...");
        client
            .batch_execute(&sql)
            .await
            .with_context(|| format!("failed executing {}", name))?;
        println!(" - {name}: OK");
    }

    // sanity ping
    let row = client.query_one("SELECT 1", &[]).await?;
    let ok: i32 = row.get(0);
    println!("db_init: done, ping={ok}");

    Ok(())
}
