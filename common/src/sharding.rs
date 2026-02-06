use std::hash::Hasher;
use twox_hash::XxHash64;

use crate::{Symbol, Timeframe};

#[derive(Clone, Copy, Debug)]
pub enum ShardStrategy {
    /// Быстро, но при изменении shard_count “перетасовывает” почти всё.
    Mod,
    /// Стабильно при изменении shard_count (минимум миграций ключей).
    Jump,
}

#[inline]
pub fn shard_key(stream: &str, symbol: &Symbol, timeframe: Timeframe) -> String {
    // стабильный ключ: один symbol+tf всегда в одну партицию
    format!("{}|{}|{}", stream, symbol.as_str(), timeframe.as_str())
}

#[inline]
pub fn hash64(s: &str) -> u64 {
    let mut h = XxHash64::with_seed(0);
    h.write(s.as_bytes());
    h.finish()
}

/// Jump Consistent Hash (Lamping, Veach).
/// Возвращает bucket [0..buckets-1].
#[inline]
pub fn jump_consistent_hash(mut key: u64, buckets: i32) -> i32 {
    // buckets must be > 0
    let mut b: i64 = -1;
    let mut j: i64 = 0;
    while j < buckets as i64 {
        b = j;
        key = key.wrapping_mul(2862933555777941757).wrapping_add(1);
        let inv = ((key >> 33) + 1) as f64;
        j = (((b + 1) as f64) * ((1u64 << 31) as f64 / inv)) as i64;
    }
    b as i32
}

#[inline]
pub fn get_shard_id(key: &str, shard_count: usize) -> usize {
    if shard_count == 0 {
        return 0;
    }
    // По умолчанию — Jump (стабильнее для распределенных кешей/окон).
    let h = hash64(key);
    jump_consistent_hash(h, shard_count as i32) as usize
}

#[inline]
pub fn get_series_shard(symbol: &Symbol, timeframe: Timeframe, shard_count: usize) -> usize {
    let k = shard_key("default", symbol, timeframe);
    get_shard_id(&k, shard_count)
}
