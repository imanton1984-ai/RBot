pub mod migrations;

use std::path::PathBuf;

pub fn default_ddl_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ddl")
}