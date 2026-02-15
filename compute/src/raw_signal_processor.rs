// compute/src/raw_signal_processor.rs

use std::collections::HashMap;

use common::{Symbol, Timeframe};
use serde_json::{json, Map, Value as JsonValue};

use crate::compute_backend::FeatureWindow;
use crate::raw_signal_types::RawSignal as ComputeRawSignal;


use raw_signals::thresholds::{RawSignal as RawSignalsRawSignal, SignalConfig};

pub struct RawSignalProcessor {
    pub config: SignalConfig,
}

impl RawSignalProcessor {
    pub fn new(config: SignalConfig) -> Self {
        Self { config }
    }

    pub fn process_feature_window(&self, feature_window: &FeatureWindow) -> Vec<ComputeRawSignal> {
        // раньше ты строил feat_idx по feature_window.features
        // теперь читаем сразу из feature_window.batch + свечей

        let cw = feature_window.candle_window.as_ref().expect("No candle window available");
        let n = feature_window.batch.len();
        if n == 0 { return vec![]; }

        // Эти два вектора создаём ОДИН раз на окно (raw_signals API пока требует массивы)
        let symbols = vec![feature_window.symbol.clone(); n];
        let timeframes = vec![feature_window.timeframe; n];

        // Вытаскиваем колонки без выделений/хешмап на бар
        let adx = feature_window.batch.get_f64("adx").unwrap_or(&[]);
        let atr = feature_window.batch.get_f64("atr").unwrap_or(&[]);
        let rsi = feature_window.batch.get_f64("rsi").unwrap_or(&[]);
        let cci = feature_window.batch.get_f64("cci").unwrap_or(&[]);
        let ema_20 = feature_window.batch.get_f64("ema_20").unwrap_or(&[]);
        let ema_50 = feature_window.batch.get_f64("ema_50").unwrap_or(&[]);
        let ema_200 = feature_window.batch.get_f64("ema_200").unwrap_or(&[]);
        let sma = feature_window.batch.get_f64("sma").unwrap_or(&[]);
        let obv = feature_window.batch.get_f64("obv").unwrap_or(&[]);
        let vwap = feature_window.batch.get_f64("vwap").unwrap_or(&[]);
        let williams = feature_window.batch.get_f64("williams").unwrap_or(&[]);
        let bb_upper = feature_window.batch.get_f64("bb_upper").unwrap_or(&[]);
        let bb_mid = feature_window.batch.get_f64("bb_mid").unwrap_or(&[]);
        let bb_lower = feature_window.batch.get_f64("bb_lower").unwrap_or(&[]);
        let _macd = feature_window.batch.get_f64("macd").unwrap_or(&[]);
        let _macd_signal = feature_window.batch.get_f64("macd_signal").unwrap_or(&[]);
        let macd_hist = feature_window.batch.get_f64("macd_hist").unwrap_or(&[]);
        let stoch_k = feature_window.batch.get_f64("stoch_k").unwrap_or(&[]);
        let stoch_d = feature_window.batch.get_f64("stoch_d").unwrap_or(&[]);
        let volume_spike = feature_window.batch.get_f64("volume_spike").unwrap_or(&[]);

        let timestamps = feature_window.batch.timestamps.as_slice();

        let mut raw_signals: Vec<ComputeRawSignal> = Vec::new();

        // Process each indicator separately to minimize memory usage
        raw_signals.extend(
            raw_signals::adx_raw::calculate_adx_raw_signals(adx, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        raw_signals.extend(
            raw_signals::atr_raw::calculate_atr_raw_signals(atr, &cw.close, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        raw_signals.extend(
            raw_signals::bb_raw::calculate_bb_raw_signals(
                &cw.close,
                bb_upper,
                bb_mid,
                bb_lower,
                timestamps,
                &symbols,
                &timeframes,
                &self.config,
            )
            .into_iter()
            .map(|s| self.convert_raw_signal(s)),
        );

        raw_signals.extend(
            raw_signals::cci_raw::calculate_cci_raw_signals(cci, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        raw_signals.extend(
            raw_signals::ema_raw::calculate_ema_raw_signals(&cw.close, ema_20, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal_with_sub_id(s, 20)),
        );
        raw_signals.extend(
            raw_signals::ema_raw::calculate_ema_raw_signals(&cw.close, ema_50, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal_with_sub_id(s, 50)),
        );
        raw_signals.extend(
            raw_signals::ema_raw::calculate_ema_raw_signals(&cw.close, ema_200, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal_with_sub_id(s, 200)),
        );

        raw_signals.extend(
            raw_signals::macd_raw::calculate_macd_raw_signals(macd_hist, &cw.close, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        raw_signals.extend(
            raw_signals::obv_raw::calculate_obv_raw_signals(obv, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        raw_signals.extend(
            raw_signals::rsi_raw::calculate_rsi_raw_signals(rsi, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        raw_signals.extend(
            raw_signals::sma_raw::calculate_sma_raw_signals(&cw.close, sma, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        raw_signals.extend(
            raw_signals::stoch_raw::calculate_stoch_raw_signals(stoch_k, stoch_d, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        raw_signals.extend(
            raw_signals::vwap_raw::calculate_vwap_raw_signals(&cw.close, vwap, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        raw_signals.extend(
            raw_signals::williams_raw::calculate_williams_raw_signals(williams, timestamps, &symbols, &timeframes, &self.config)
                .into_iter()
                .map(|s| self.convert_raw_signal(s)),
        );

        raw_signals.extend(
            raw_signals::volume_spike_raw::calculate_volume_spike_raw_signals(
                volume_spike, // Now passing f64 slice
                &cw.volume,
                timestamps,
                &symbols,
                &timeframes,
                &self.config,
            )
            .into_iter()
            .map(|s| self.convert_raw_signal(s)),
        );

        // Приклеиваем контекст (features_json + scores_json) по свече.
        self.attach_context(
            &mut raw_signals,
            cw,
            &feature_window.batch,
            &symbols,
            &timeframes,
        );

        raw_signals
    }

    // Updated attach_context function without (Symbol,Timeframe,ts) keys
    fn attach_context(
        &self,
        signals: &mut [ComputeRawSignal],
        cw: &crate::CandleWindow,
        batch: &crate::FeatureBatch,
        symbols: &[Symbol],
        timeframes: &[Timeframe],
    ) {
        let mut idx = std::collections::HashMap::<i64, usize>::with_capacity(cw.timestamps.len());
        for (i, &ts) in cw.timestamps.iter().enumerate() {
            idx.insert(ts, i);
        }

        let col = |name: &str, i: usize| -> f64 {
            batch.get_f64(name).and_then(|v| v.get(i).copied()).unwrap_or(f64::NAN)
        };

        let mut groups: HashMap<(common::Symbol, common::Timeframe, i64), Vec<usize>> = HashMap::new();
        for (i, s) in signals.iter().enumerate() {
            groups.entry((s.symbol.clone(), s.timeframe, s.timestamp)).or_default().push(i);
        }

        for (k, idxs) in groups {
            let Some(&ci) = idx.get(&k.2) else { continue; }; // k.2 is timestamp

            let mut fm = Map::new();
            fm.insert("symbol".into(), json!(symbols[ci].to_string()));
            fm.insert("tf".into(), json!(timeframes[ci].as_str()));
            fm.insert("time_ms".into(), json!(cw.timestamps[ci]));
            fm.insert("close".into(), json!(cw.close[ci]));
            fm.insert("high".into(), json!(cw.high[ci]));
            fm.insert("low".into(), json!(cw.low[ci]));
            fm.insert("volume".into(), json!(cw.volume[ci]));

            put_num(&mut fm, "adx", col("adx", ci));
            put_num(&mut fm, "atr", col("atr", ci));
            put_num(&mut fm, "bb_upper", col("bb_upper", ci));
            put_num(&mut fm, "bb_mid", col("bb_mid", ci));
            put_num(&mut fm, "bb_lower", col("bb_lower", ci));
            put_num(&mut fm, "cci", col("cci", ci));
            put_num(&mut fm, "ema_20", col("ema_20", ci));
            put_num(&mut fm, "ema_50", col("ema_50", ci));
            put_num(&mut fm, "ema_200", col("ema_200", ci));
            put_num(&mut fm, "macd", col("macd", ci));
            put_num(&mut fm, "macd_signal", col("macd_signal", ci));
            put_num(&mut fm, "macd_hist", col("macd_hist", ci));
            put_num(&mut fm, "obv", col("obv", ci));
            put_num(&mut fm, "rsi", col("rsi", ci));
            put_num(&mut fm, "sma", col("sma", ci));
            put_num(&mut fm, "stoch_k", col("stoch_k", ci));
            put_num(&mut fm, "stoch_d", col("stoch_d", ci));
            put_num(&mut fm, "vwap", col("vwap", ci));
            put_num(&mut fm, "williams", col("williams", ci));
            put_num(&mut fm, "volume_spike", col("volume_spike", ci));

            let features_json = JsonValue::Object(fm);

            let mut arr = Vec::with_capacity(idxs.len());
            for &si in &idxs {
                let s = &signals[si];
                arr.push(json!({
                    "indicator_id": s.indicator_id,
                    "signal_kind": s.signal_kind,
                    "signal_sub_id": s.signal_sub_id,
                    "side": s.side,
                    "score": s.score,
                    "value": s.value
                }));
            }
            let scores_json = json!({ "signals": arr });

            // Attach JSON only to the first signal of the candle group
            if let Some(&first_si) = idxs.get(0) {
                signals[first_si].features_json = Some(features_json);
                signals[first_si].scores_json = Some(scores_json);
            }

            // Set metadata for all signals in the group
            for &si in &idxs {
                signals[si].candle_is_final = true;
                signals[si].calc_source = 1;
            }
        }
    }

    fn convert_raw_signal(&self, raw_signal: RawSignalsRawSignal) -> ComputeRawSignal {
        ComputeRawSignal {
            symbol: raw_signal.symbol,
            timeframe: raw_signal.timeframe,
            timestamp: raw_signal.timestamp,
            indicator_id: raw_signal.indicator_id,
            signal_kind: raw_signal.signal_kind,
            signal_sub_id: 0, // default (если захочешь - можно маппить по details)
            side: raw_signal.side,
            score: raw_signal.score,
            value: raw_signal.value,
            details: raw_signal.details,

            candle_is_final: true,
            calc_source: 1,
            event_time_ms: None,

            features_json: None,
            scores_json: None,
            predictors_json: None,
        }
    }

    fn convert_raw_signal_with_sub_id(&self, raw_signal: RawSignalsRawSignal, sub_id: i16) -> ComputeRawSignal {
        let mut s = self.convert_raw_signal(raw_signal);
        s.signal_sub_id = sub_id;
        s
    }
}

fn put_num(m: &mut Map<String, JsonValue>, k: &str, v: f64) {
    if v.is_finite() {
        m.insert(k.to_string(), json!(v));
    }
}