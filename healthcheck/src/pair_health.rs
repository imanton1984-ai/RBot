use sqlx::PgPool;

pub struct PairHealthChecker;

pub struct PairHealthResult {
    pub ok: bool,
    pub active_pairs: i64,
    pub last_refreshed_age_sec: Option<i64>,
}

impl PairHealthChecker {
    pub async fn check(db_url: &str, min_pairs: i64, max_age_sec: i64) -> PairHealthResult {
        let pool = match PgPool::connect(db_url).await {
            Ok(p) => p,
            Err(_) => {
                return PairHealthResult { ok: false, active_pairs: 0, last_refreshed_age_sec: None };
            }
        };

        let (active_pairs,): (i64,) = match sqlx::query_as(
            "SELECT COUNT(*) FROM market.pairs WHERE is_active = TRUE"
        ).fetch_one(&pool).await {
            Ok(v) => v,
            Err(_) => return PairHealthResult { ok: false, active_pairs: 0, last_refreshed_age_sec: None },
        };

        let (age_sec_opt,): (Option<i64>,) = match sqlx::query_as(
            "SELECT EXTRACT(EPOCH FROM (now() - MAX(last_refreshed_at)))::bigint FROM market.pairs"
        ).fetch_one(&pool).await {
            Ok(v) => v,
            Err(_) => (None,),
        };

        let age_ok = age_sec_opt.map(|age| age <= max_age_sec).unwrap_or(false);
        let ok = active_pairs >= min_pairs && age_ok;

        PairHealthResult { ok, active_pairs, last_refreshed_age_sec: age_sec_opt }
    }
}
