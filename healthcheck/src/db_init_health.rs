use sqlx::{PgPool, postgres::PgConnectOptions};

pub struct DbInitHealthChecker;

impl DbInitHealthChecker {
    pub async fn check_db_connectivity(&self, db_url: &str) -> bool {
        let opts: PgConnectOptions = db_url.parse().expect("Invalid DB URL");

        match PgPool::connect_with(opts).await {
            Ok(_pool) => true,
            Err(_) => false,
        }
    }
}