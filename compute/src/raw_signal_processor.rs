use crate::{FeatureWindow, RawSignal as ComputeRawSignal};
use raw_signals::thresholds::{SignalConfig, RawSignal as RawSignalsRawSignal};
use common::{Symbol, Timeframe};
use std::vec::Vec;

pub struct RawSignalProcessor {
    config: SignalConfig,
}

impl RawSignalProcessor {
    pub fn new(config: SignalConfig) -> Self {
        Self { config }
    }

    pub fn process_feature_window(&self, feature_window: &FeatureWindow) -> Vec<ComputeRawSignal> {
        let mut raw_signals = Vec::new();

        if feature_window.features.is_empty() {
            return raw_signals;
        }

        let candle_window = match &feature_window.candle_window {
            Some(cw) => cw,
            None => return raw_signals,
        };

        let timestamps: Vec<i64> = candle_window.timestamps.clone();
        let symbols: Vec<Symbol> = vec![feature_window.features[0].symbol.clone(); timestamps.len()];
        let timeframes: Vec<Timeframe> = vec![feature_window.features[0].timeframe; timestamps.len()];

        // poc
        let poc_values = feature_window.get_indicator_values("poc");
        let poc_raw_signals = raw_signals::poc_raw::calculate_poc_raw_signals(&candle_window.close, &poc_values, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(poc_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // adx
        let adx_values = feature_window.get_indicator_values("adx");
        let adx_raw_signals = raw_signals::adx_raw::calculate_adx_raw_signals(&adx_values, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(adx_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // atr
        let atr_values = feature_window.get_indicator_values("atr");
        let atr_raw_signals = raw_signals::atr_raw::calculate_atr_raw_signals(&atr_values, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(atr_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // bb
        let bb_upper = feature_window.get_indicator_values("bb_upper");
        let bb_mid = feature_window.get_indicator_values("bb_mid");
        let bb_lower = feature_window.get_indicator_values("bb_lower");
        let bb_raw_signals = raw_signals::bb_raw::calculate_bb_raw_signals(&candle_window.close, &bb_upper, &bb_mid, &bb_lower, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(bb_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // cci
        let cci_values = feature_window.get_indicator_values("cci");
        let cci_raw_signals = raw_signals::cci_raw::calculate_cci_raw_signals(&cci_values, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(cci_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // ema20
        let ema20 = feature_window.get_indicator_values("ema20");
        let ema20_raw_signals = raw_signals::ema_raw::calculate_ema_raw_signals(&candle_window.close, &ema20, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(ema20_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // ema50
        let ema50 = feature_window.get_indicator_values("ema50");
        let ema50_raw_signals = raw_signals::ema_raw::calculate_ema_raw_signals(&candle_window.close, &ema50, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(ema50_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // ema200
        let ema200 = feature_window.get_indicator_values("ema200");
        let ema200_raw_signals = raw_signals::ema_raw::calculate_ema_raw_signals(&candle_window.close, &ema200, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(ema200_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // macd
        let macd_line = feature_window.get_indicator_values("macd");
        let macd_signal = feature_window.get_indicator_values("macd_signal");
        let macd_hist = feature_window.get_indicator_values("macd_hist");
        let macd_cross_signals = raw_signals::macd_raw::calculate_macd_crossover_signals(&macd_line, &macd_signal, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(macd_cross_signals.into_iter().map(|s| self.convert_raw_signal(s)));
        let macd_hist_signals = raw_signals::macd_raw::calculate_macd_histogram_signals(&macd_hist, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(macd_hist_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // obv
        let obv_values = feature_window.get_indicator_values("obv");
        let obv_raw_signals = raw_signals::obv_raw::calculate_obv_raw_signals(&obv_values, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(obv_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // rsi
        let rsi_values = feature_window.get_indicator_values("rsi");
        let rsi_raw_signals = raw_signals::rsi_raw::calculate_rsi_raw_signals(&rsi_values, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(rsi_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // sma
        let sma_values = feature_window.get_indicator_values("sma");
        let sma_raw_signals = raw_signals::sma_raw::calculate_sma_raw_signals(&candle_window.close, &sma_values, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(sma_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // stoch
        let k_values = feature_window.get_indicator_values("stoch_k");
        let d_values = feature_window.get_indicator_values("stoch_d");
        let stoch_raw_signals = raw_signals::stoch_raw::calculate_stoch_raw_signals(&k_values, &d_values, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(stoch_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // vwap
        let vwap_values = feature_window.get_indicator_values("vwap");
        let vwap_raw_signals = raw_signals::vwap_raw::calculate_vwap_raw_signals(&candle_window.close, &vwap_values, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(vwap_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        // williams
        let williams_values = feature_window.get_indicator_values("williams");
        let williams_raw_signals = raw_signals::williams_raw::calculate_williams_raw_signals(&williams_values, &timestamps, &symbols, &timeframes, &self.config);
        raw_signals.extend(williams_raw_signals.into_iter().map(|s| self.convert_raw_signal(s)));

        raw_signals
    }

    fn convert_raw_signal(&self, raw_signal: RawSignalsRawSignal) -> ComputeRawSignal {
        ComputeRawSignal {
            symbol: raw_signal.symbol,
            timeframe: raw_signal.timeframe,
            timestamp: raw_signal.timestamp,
            indicator_id: raw_signal.indicator_id,
            signal_kind: raw_signal.signal_kind,
            side: raw_signal.side,
            score: raw_signal.score,
            value: raw_signal.value,
            details: raw_signal.details,
        }
    }
}
