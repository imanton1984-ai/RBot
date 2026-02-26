// order_manager/src/tests_synthetic.rs
//
// Синтетические тесты для критических операций:
//   - Открытие LONG/SHORT позиций на фьючерсном рынке
//   - Закрытие по Take Profit (TP)
//   - Закрытие по Stop Loss (SL)
//   - Закрытие по истечению candles_left (Max Bars)
//   - Расчёт PnL (без плеча — реальный USDT)
//   - Расчёт комиссий
//   - Расчёт количества (stepSize, minNotional)
//   - Логика пропорций таймфреймов 1h:4h:15m
//   - Валидация сигналов (score range, price drift)
//   - Emergency Close (принудительное закрытие)
//   - Проверка candles_left декремента

#[cfg(test)]
mod synthetic_futures_tests {
    use crate::types::*;
    use crate::config::OrderManagerConfig;
    use chrono::Utc;

    // ─── Хелпер для создания тестовой позиции ──────────────────────────
    fn make_long_position(
        id: i64,
        symbol: &str,
        entry: f64,
        qty: f64,
        leverage: u16,
        sl: f64,
        tp: f64,
        candles_left: i16,
    ) -> ManagedPosition {
        ManagedPosition {
            position_id: id,
            symbol: symbol.to_string(),
            symbol_id: 1,
            tf_minutes: 60,
            side: Side::Long,
            entry_price: entry,
            sl_price: sl,
            tp_price: tp,
            qty,
            leverage,
            candles_left,
            max_hold_bars: 25,
            unrealized_pnl: 0.0,
            unrealized_pnl_pct: 0.0,
            current_price: entry,
            opened_at: Utc::now(),
            combined_score: 0.75,
            p_super: 0.8,
            entry_order_id: Some("test-entry-1".to_string()),
            sl_order_id: Some("test-sl-1".to_string()),
            tp_order_id: Some("test-tp-1".to_string()),
        }
    }

    fn make_short_position(
        id: i64,
        symbol: &str,
        entry: f64,
        qty: f64,
        leverage: u16,
        sl: f64,
        tp: f64,
        candles_left: i16,
    ) -> ManagedPosition {
        ManagedPosition {
            position_id: id,
            symbol: symbol.to_string(),
            symbol_id: 1,
            tf_minutes: 60,
            side: Side::Short,
            entry_price: entry,
            sl_price: sl,
            tp_price: tp,
            qty,
            leverage,
            candles_left,
            max_hold_bars: 25,
            unrealized_pnl: 0.0,
            unrealized_pnl_pct: 0.0,
            current_price: entry,
            opened_at: Utc::now(),
            combined_score: 0.72,
            p_super: 0.65,
            entry_order_id: Some("test-entry-2".to_string()),
            sl_order_id: Some("test-sl-2".to_string()),
            tp_order_id: Some("test-tp-2".to_string()),
        }
    }

    fn make_signal(symbol: &str, side: Side, price: f64, sl: f64, tp: f64, score: f32) -> QualifiedSignal {
        QualifiedSignal {
            signal_time: Utc::now(),
            signal_time_ms: Utc::now().timestamp_millis(),
            symbol: symbol.to_string(),
            symbol_id: 1,
            tf_minutes: 60,
            side,
            entry_price: price,
            sl_price: sl,
            tp_price: tp,
            p_super: 0.8,
            combined_score: score,
            current_price: price,
            price_drift_pct: 0.0,
        }
    }

    // ═══════════════════════════════════════════════════════════
    // 1. OPEN POSITION — Quantity Calculation
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_quantity_calculation_btc() {
        // trade_size = 100 USDT, leverage = 10x, BTC price = 50000
        // notional = 100 * 10 = 1000 USDT
        // qty = 1000 / 50000 = 0.02 BTC
        let trade_size: f64 = 100.0;
        let leverage: u16 = 10;
        let price: f64 = 50000.0;
        let notional = trade_size * leverage as f64;
        let qty = notional / price;
        assert!((qty - 0.02).abs() < 1e-10, "BTC qty should be 0.02, got {}", qty);
    }

    #[test]
    fn test_quantity_calculation_eth() {
        // trade_size = 100 USDT, leverage = 10x, ETH price = 3000
        // notional = 1000 USDT
        // qty = 1000 / 3000 ≈ 0.33333
        let trade_size: f64 = 100.0;
        let leverage: u16 = 10;
        let price: f64 = 3000.0;
        let notional = trade_size * leverage as f64;
        let qty = notional / price;
        assert!((qty - 0.333333).abs() < 0.001, "ETH qty should be ~0.333, got {}", qty);
    }

    #[test]
    fn test_quantity_calculation_low_price_alt() {
        // trade_size = 100 USDT, leverage = 10x, DOGE price = 0.08
        // notional = 1000 USDT
        // qty = 1000 / 0.08 = 12500 DOGE
        let trade_size: f64 = 100.0;
        let leverage: u16 = 10;
        let price: f64 = 0.08;
        let notional = trade_size * leverage as f64;
        let qty = notional / price;
        assert!((qty - 12500.0).abs() < 0.001, "DOGE qty should be 12500, got {}", qty);
    }

    #[test]
    fn test_quantity_step_size_rounding() {
        // BTC stepSize = 0.001 (3 decimals)
        let raw_qty: f64 = 0.02345678;
        let step_size: f64 = 0.001;
        let rounded = (raw_qty / step_size).floor() * step_size;
        assert!((rounded - 0.023).abs() < 1e-10, "Rounded qty should be 0.023, got {}", rounded);
    }

    #[test]
    fn test_min_notional_check() {
        // Binance Futures minNotional = 5 USDT
        let qty: f64 = 0.0001; // BTC
        let price: f64 = 50000.0;
        let notional = qty * price; // 5.0 USDT
        let min_notional: f64 = 5.0;
        assert!(notional >= min_notional, "Notional {} should be >= {}", notional, min_notional);

        let qty_too_small: f64 = 0.00001;
        let notional_too_small = qty_too_small * price; // 0.5 USDT
        assert!(notional_too_small < min_notional, "Should be below minNotional");
    }

    // ═══════════════════════════════════════════════════════════
    // 2. PNL CALCULATION (without leverage = real USDT PnL)
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_pnl_long_profit() {
        // LONG BTC: entry=50000, exit=51000, qty=0.02
        // PnL = 1 * (51000 - 50000) * 0.02 = 20 USDT
        // PnL% = 1 * (51000 - 50000) / 50000 * 100 = 2%
        let pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20);
        let (pnl, pnl_pct) = pos.calc_unrealized_pnl(51000.0);
        assert!((pnl - 20.0).abs() < 0.01, "Long profit should be 20 USDT, got {}", pnl);
        assert!((pnl_pct - 2.0).abs() < 0.01, "Long profit% should be 2%, got {}", pnl_pct);
    }

    #[test]
    fn test_pnl_long_loss() {
        // LONG BTC: entry=50000, exit=49500, qty=0.02
        // PnL = 1 * (49500 - 50000) * 0.02 = -10 USDT
        let pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20);
        let (pnl, pnl_pct) = pos.calc_unrealized_pnl(49500.0);
        assert!((pnl - (-10.0)).abs() < 0.01, "Long loss should be -10 USDT, got {}", pnl);
        assert!((pnl_pct - (-1.0)).abs() < 0.01, "Long loss% should be -1%, got {}", pnl_pct);
    }

    #[test]
    fn test_pnl_short_profit() {
        // SHORT ETH: entry=3000, exit=2900, qty=1.0
        // PnL = -1 * (2900 - 3000) * 1.0 = 100 USDT (profit!)
        let pos = make_short_position(2, "ETHUSDT", 3000.0, 1.0, 10, 3100.0, 2800.0, 20);
        let (pnl, pnl_pct) = pos.calc_unrealized_pnl(2900.0);
        assert!((pnl - 100.0).abs() < 0.01, "Short profit should be 100 USDT, got {}", pnl);
        assert!((pnl_pct - 3.333).abs() < 0.1, "Short profit% should be ~3.33%, got {}", pnl_pct);
    }

    #[test]
    fn test_pnl_short_loss() {
        // SHORT ETH: entry=3000, exit=3050, qty=1.0
        // PnL = -1 * (3050 - 3000) * 1.0 = -50 USDT (loss)
        let pos = make_short_position(2, "ETHUSDT", 3000.0, 1.0, 10, 3100.0, 2800.0, 20);
        let (pnl, _) = pos.calc_unrealized_pnl(3050.0);
        assert!((pnl - (-50.0)).abs() < 0.01, "Short loss should be -50 USDT, got {}", pnl);
    }

    #[test]
    fn test_pnl_is_real_not_leveraged() {
        // Критический тест: PnL не должен быть умножен на плечо!
        // LONG BTC: entry=50000, exit=51000, qty=0.02, leverage=10x
        // Реальный PnL = (51000-50000) * 0.02 = 20 USDT
        // ЛОЖНЫЙ PnL с плечом = 20 * 10 = 200 USDT (НЕПРАВИЛЬНО!)
        let pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20);
        let (pnl, _) = pos.calc_unrealized_pnl(51000.0);

        // PnL MUST be 20, NOT 200
        assert!((pnl - 20.0).abs() < 0.01, "PnL should be 20 USDT (real), NOT {} (leveraged?)", pnl);
        assert!((pnl - 200.0).abs() > 1.0, "PnL should NOT be 200 (leveraged). Got {}", pnl);
    }

    // Дополнительный тест: margin used (без плеча)
    #[test]
    fn test_margin_calculation() {
        // trade_size = 100 USDT, leverage = 10x
        // Notional on exchange = 1000 USDT (leveraged)
        // Margin used (real money) = 100 USDT
        // Overall balance = 500 USDT
        // In Orders = 100 USDT (not 1000!)
        let trade_size: f64 = 100.0;
        let leverage: u16 = 10;
        let notional = trade_size * leverage as f64;
        let margin_used = notional / leverage as f64; // = 100 USDT

        assert!((margin_used - 100.0).abs() < 0.01, "Margin should be 100 USDT, got {}", margin_used);
        assert!((margin_used - 1000.0).abs() > 1.0, "Margin should NOT be 1000 (notional)");
    }

    // ═══════════════════════════════════════════════════════════
    // 3. STOP LOSS TRIGGERS
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_sl_hit_long() {
        // LONG: entry=50000, SL=49000
        // Price drops to 49000 → SL triggered
        let pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20);

        assert!(!pos.is_sl_hit(49001.0), "SL should NOT trigger above 49000");
        assert!(pos.is_sl_hit(49000.0), "SL should trigger at 49000");
        assert!(pos.is_sl_hit(48500.0), "SL should trigger below 49000");
    }

    #[test]
    fn test_sl_hit_short() {
        // SHORT: entry=3000, SL=3100
        // Price rises to 3100 → SL triggered
        let pos = make_short_position(2, "ETHUSDT", 3000.0, 1.0, 10, 3100.0, 2800.0, 20);

        assert!(!pos.is_sl_hit(3099.0), "SL should NOT trigger below 3100");
        assert!(pos.is_sl_hit(3100.0), "SL should trigger at 3100");
        assert!(pos.is_sl_hit(3200.0), "SL should trigger above 3100");
    }

    #[test]
    fn test_sl_pnl_long() {
        // LONG: entry=50000, SL=49000, qty=0.02
        // SL PnL = (49000 - 50000) * 0.02 = -20 USDT
        let pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20);
        let (pnl, _) = pos.calc_unrealized_pnl(49000.0);
        assert!((pnl - (-20.0)).abs() < 0.01, "Long SL PnL should be -20 USDT, got {}", pnl);
    }

    #[test]
    fn test_sl_pnl_short() {
        // SHORT: entry=3000, SL=3100, qty=1.0
        // SL PnL = -1 * (3100 - 3000) * 1.0 = -100 USDT
        let pos = make_short_position(2, "ETHUSDT", 3000.0, 1.0, 10, 3100.0, 2800.0, 20);
        let (pnl, _) = pos.calc_unrealized_pnl(3100.0);
        assert!((pnl - (-100.0)).abs() < 0.01, "Short SL PnL should be -100 USDT, got {}", pnl);
    }

    // ═══════════════════════════════════════════════════════════
    // 4. TAKE PROFIT TRIGGERS
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_tp_hit_long() {
        // LONG: entry=50000, TP=52000
        // Price rises to 52000 → TP triggered
        let pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20);

        assert!(!pos.is_tp_hit(51999.0), "TP should NOT trigger below 52000");
        assert!(pos.is_tp_hit(52000.0), "TP should trigger at 52000");
        assert!(pos.is_tp_hit(53000.0), "TP should trigger above 52000");
    }

    #[test]
    fn test_tp_hit_short() {
        // SHORT: entry=3000, TP=2800
        // Price drops to 2800 → TP triggered
        let pos = make_short_position(2, "ETHUSDT", 3000.0, 1.0, 10, 3100.0, 2800.0, 20);

        assert!(!pos.is_tp_hit(2801.0), "TP should NOT trigger above 2800");
        assert!(pos.is_tp_hit(2800.0), "TP should trigger at 2800");
        assert!(pos.is_tp_hit(2700.0), "TP should trigger below 2800");
    }

    #[test]
    fn test_tp_pnl_long() {
        // LONG: entry=50000, TP=52000, qty=0.02
        // TP PnL = (52000 - 50000) * 0.02 = 40 USDT
        let pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20);
        let (pnl, pnl_pct) = pos.calc_unrealized_pnl(52000.0);
        assert!((pnl - 40.0).abs() < 0.01, "Long TP PnL should be 40 USDT, got {}", pnl);
        assert!((pnl_pct - 4.0).abs() < 0.01, "Long TP PnL% should be 4%, got {}", pnl_pct);
    }

    #[test]
    fn test_tp_pnl_short() {
        // SHORT: entry=3000, TP=2800, qty=1.0
        // TP PnL = -1 * (2800 - 3000) * 1.0 = 200 USDT
        let pos = make_short_position(2, "ETHUSDT", 3000.0, 1.0, 10, 3100.0, 2800.0, 20);
        let (pnl, _) = pos.calc_unrealized_pnl(2800.0);
        assert!((pnl - 200.0).abs() < 0.01, "Short TP PnL should be 200 USDT, got {}", pnl);
    }

    // ═══════════════════════════════════════════════════════════
    // 5. CANDLES LEFT / MAX BARS EXPIRY
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_candles_left_expiry() {
        let mut pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 25);
        assert!(!pos.is_expired());

        // Simulate candle ticks
        pos.candles_left = 1;
        assert!(!pos.is_expired());

        pos.candles_left = 0;
        assert!(pos.is_expired());

        pos.candles_left = -1; // Shouldn't happen but test safety
        assert!(pos.is_expired());
    }

    #[test]
    fn test_candles_left_decrement() {
        let mut pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 25);

        // Tick 1 candle
        pos.candles_left -= 1;
        assert_eq!(pos.candles_left, 24);

        // Tick 5 candles (batch)
        pos.candles_left = (pos.candles_left - 5).max(0);
        assert_eq!(pos.candles_left, 19);

        // Duration lived calculation
        let bars_lived = pos.max_hold_bars - pos.candles_left;
        assert_eq!(bars_lived, 6, "Should have lived 6 bars");
    }

    #[test]
    fn test_max_bars_force_close_pnl() {
        // Position expired with slight profit
        let pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 0);
        let exit_price = 50500.0;
        let (pnl, _) = pos.calc_unrealized_pnl(exit_price);
        // PnL = (50500 - 50000) * 0.02 = 10 USDT
        assert!((pnl - 10.0).abs() < 0.01, "Force close PnL should be 10 USDT, got {}", pnl);
        assert!(pos.is_expired(), "Position should be expired");
    }

    // ═══════════════════════════════════════════════════════════
    // 6. FEE CALCULATIONS
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_taker_fee_calculation() {
        // Binance Futures taker fee = 0.04%
        let taker_fee_pct: f64 = 0.04;
        let entry_price: f64 = 50000.0;
        let exit_price: f64 = 51000.0;
        let qty: f64 = 0.02;

        let entry_notional = entry_price * qty; // 1000 USDT
        let exit_notional = exit_price * qty;   // 1020 USDT
        let fees = (entry_notional + exit_notional) * taker_fee_pct / 100.0;

        // Fees = (1000 + 1020) * 0.0004 = 0.808 USDT
        assert!((fees - 0.808).abs() < 0.001, "Fees should be ~0.808, got {}", fees);
    }

    #[test]
    fn test_net_pnl_after_fees() {
        let entry: f64 = 50000.0;
        let exit: f64 = 51000.0;
        let qty: f64 = 0.02;
        let taker_fee_pct: f64 = 0.04;

        let gross_pnl = (exit - entry) * qty; // 20 USDT
        let fees = ((entry * qty) + (exit * qty)) * taker_fee_pct / 100.0; // ~0.808
        let net_pnl = gross_pnl - fees;

        assert!((net_pnl - 19.192).abs() < 0.01, "Net PnL should be ~19.19, got {}", net_pnl);
        assert!(net_pnl > 0.0, "Should still be profitable after fees");
    }

    #[test]
    fn test_fee_eats_small_profit() {
        // Tiny move: entry=50000, exit=50005, qty=0.02
        let entry: f64 = 50000.0;
        let exit: f64 = 50005.0;
        let qty: f64 = 0.02;
        let taker_fee_pct: f64 = 0.04;

        let gross_pnl = (exit - entry) * qty; // 0.1 USDT
        let fees = ((entry * qty) + (exit * qty)) * taker_fee_pct / 100.0; // ~0.8
        let net_pnl = gross_pnl - fees;

        assert!(net_pnl < 0.0, "Small move should be net-negative due to fees. Net: {}", net_pnl);
    }

    // ═══════════════════════════════════════════════════════════
    // 7. CLOSE REASON ENUM
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_close_reasons() {
        assert_eq!(CloseReason::TpHit.as_str(), "tp_hit");
        assert_eq!(CloseReason::SlHit.as_str(), "sl_hit");
        assert_eq!(CloseReason::MaxBars.as_str(), "max_bars");
        assert_eq!(CloseReason::RiskManager.as_str(), "risk_manager");
        assert_eq!(CloseReason::Manual.as_str(), "manual");
    }

    // ═══════════════════════════════════════════════════════════
    // 8. TIMEFRAME ALLOCATION
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_timeframe_allocation_default() {
        let alloc = TimeframeAllocation::default_10();
        assert_eq!(alloc.total, 10);
        assert_eq!(alloc.slots_for_tf(60), 7);   // 1h = 70%
        assert_eq!(alloc.slots_for_tf(240), 2);  // 4h = 20%
        assert_eq!(alloc.slots_for_tf(15), 1);   // 15m = 10%
        assert_eq!(alloc.slots_for_tf(5), 0);    // untradeable TF
    }

    #[test]
    fn test_timeframe_allocation_from_config() {
        let config = OrderManagerConfig::default();
        let alloc = config.timeframe_allocation();
        assert_eq!(alloc.total, 10);

        // Ensure proportions: sum of slots = total
        let sum: u16 = alloc.slots.iter().map(|(_, count)| *count).sum();
        assert_eq!(sum, alloc.total, "Sum of TF slots must equal total");
    }

    #[test]
    fn test_timeframe_allocation_custom() {
        let mut config = OrderManagerConfig::default();
        config.max_orders_at_a_time = 20;
        // 70% 1h = 14, 20% 4h = 4, 10% 15m = 2
        let alloc = config.timeframe_allocation();
        assert_eq!(alloc.slots_for_tf(60), 14);
        assert_eq!(alloc.slots_for_tf(240), 4);
        assert_eq!(alloc.slots_for_tf(15), 2);
    }

    // ═══════════════════════════════════════════════════════════
    // 9. SIGNAL QUALIFICATION (SCORE + PRICE DRIFT)
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_signal_score_in_range() {
        let config = OrderManagerConfig::default();
        let signal = make_signal("BTCUSDT", Side::Long, 50000.0, 49000.0, 52000.0, 0.75);
        assert!(signal.combined_score >= config.signal_score_min_1h);
        assert!(signal.combined_score <= config.signal_score_max_1h);
    }

    #[test]
    fn test_signal_score_below_range() {
        let config = OrderManagerConfig::default();
        let signal = make_signal("BTCUSDT", Side::Long, 50000.0, 49000.0, 52000.0, 0.60);
        assert!(signal.combined_score < config.signal_score_min_1h,
            "Signal with score 0.60 should be below min threshold 0.70");
    }

    #[test]
    fn test_signal_score_above_range() {
        let config = OrderManagerConfig::default();
        let signal = make_signal("BTCUSDT", Side::Long, 50000.0, 49000.0, 52000.0, 0.90);
        assert!(signal.combined_score > config.signal_score_max_1h,
            "Signal with score 0.90 should be above max threshold 0.80");
    }

    #[test]
    fn test_signal_price_drift() {
        let config = OrderManagerConfig::default();
        let mut signal = make_signal("BTCUSDT", Side::Long, 50000.0, 49000.0, 52000.0, 0.75);

        // Current price matches entry → drift = 0%
        signal.current_price = 50000.0;
        signal.price_drift_pct = ((signal.current_price - signal.entry_price) / signal.entry_price).abs() * 100.0;
        assert!(signal.price_drift_pct <= config.max_price_drift_pct,
            "0% drift should be within threshold");

        // Current price drifted 0.5% → above 0.2% threshold
        signal.current_price = 50250.0;
        signal.price_drift_pct = ((signal.current_price - signal.entry_price) / signal.entry_price).abs() * 100.0;
        assert!(signal.price_drift_pct > config.max_price_drift_pct,
            "0.5% drift should be outside 0.2% threshold");
    }

    // ═══════════════════════════════════════════════════════════
    // 10. SIDE CONVERSIONS
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_side_binance_api_mapping() {
        // LONG entry = BUY, LONG close = SELL
        assert_eq!(Side::Long.entry_side_str(), "BUY");
        assert_eq!(Side::Long.close_side_str(), "SELL");

        // SHORT entry = SELL, SHORT close = BUY
        assert_eq!(Side::Short.entry_side_str(), "SELL");
        assert_eq!(Side::Short.close_side_str(), "BUY");
    }

    #[test]
    fn test_side_db_mapping() {
        assert_eq!(Side::Long.as_db_i16(), 1);
        assert_eq!(Side::Short.as_db_i16(), -1);
        assert_eq!(Side::from_db_i16(1), Some(Side::Long));
        assert_eq!(Side::from_db_i16(-1), Some(Side::Short));
        assert_eq!(Side::from_db_i16(0), None);
    }

    // ═══════════════════════════════════════════════════════════
    // 11. FULL LIFECYCLE SIMULATION
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_full_long_lifecycle_tp_hit() {
        // 1. Open LONG BTC at 50000
        let mut pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 25);

        // 2. Price moves up — still open
        pos.current_price = 51000.0;
        let (pnl, _) = pos.calc_unrealized_pnl(51000.0);
        assert!(pnl > 0.0);
        assert!(!pos.is_tp_hit(51000.0));
        assert!(!pos.is_sl_hit(51000.0));

        // 3. Tick candles
        pos.candles_left -= 5;
        assert!(!pos.is_expired());

        // 4. Price reaches TP
        pos.current_price = 52000.0;
        assert!(pos.is_tp_hit(52000.0));

        // 5. Calculate final PnL
        let (final_pnl, final_pnl_pct) = pos.calc_unrealized_pnl(52000.0);
        assert!((final_pnl - 40.0).abs() < 0.01, "Final PnL at TP should be 40 USDT");
        assert!((final_pnl_pct - 4.0).abs() < 0.01, "Final PnL% at TP should be 4%");
    }

    #[test]
    fn test_full_short_lifecycle_sl_hit() {
        // 1. Open SHORT ETH at 3000
        let mut pos = make_short_position(2, "ETHUSDT", 3000.0, 1.0, 10, 3100.0, 2800.0, 25);

        // 2. Price moves down — profit
        pos.current_price = 2950.0;
        let (pnl, _) = pos.calc_unrealized_pnl(2950.0);
        assert!(pnl > 0.0, "Should be profitable at 2950");

        // 3. Price reverses — SL hit
        pos.current_price = 3100.0;
        assert!(pos.is_sl_hit(3100.0));

        let (final_pnl, _) = pos.calc_unrealized_pnl(3100.0);
        assert!((final_pnl - (-100.0)).abs() < 0.01, "SL PnL should be -100 USDT");
    }

    #[test]
    fn test_full_lifecycle_max_bars() {
        // 1. Open LONG, price goes sideways
        let mut pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 25);

        // 2. Simulate 25 candles, price barely moved
        for _ in 0..25 {
            pos.candles_left -= 1;
        }
        assert_eq!(pos.candles_left, 0);
        assert!(pos.is_expired());

        // 3. Force close at current price (slight profit)
        pos.current_price = 50100.0;
        let (pnl, _) = pos.calc_unrealized_pnl(50100.0);
        assert!((pnl - 2.0).abs() < 0.01, "Max bars close PnL should be ~2 USDT");
    }

    // ═══════════════════════════════════════════════════════════
    // 12. EMERGENCY CLOSE
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_emergency_close_multiple_positions() {
        let positions = vec![
            make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20),
            make_short_position(2, "ETHUSDT", 3000.0, 1.0, 10, 3100.0, 2800.0, 15),
            make_long_position(3, "SOLUSDT", 100.0, 10.0, 10, 95.0, 110.0, 10),
        ];

        // Emergency = close ALL
        let to_close: Vec<_> = positions.iter().map(|p| p.position_id).collect();
        assert_eq!(to_close.len(), 3, "Emergency should close all 3 positions");
        assert_eq!(to_close, vec![1, 2, 3]);
    }

    #[test]
    fn test_risk_manager_close_specific() {
        let positions = vec![
            make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20),
            make_short_position(2, "ETHUSDT", 3000.0, 1.0, 10, 3100.0, 2800.0, 15),
            make_long_position(3, "BTCUSDT", 50500.0, 0.01, 10, 49500.0, 52500.0, 5),
        ];

        // Risk manager closes all BTCUSDT LONG
        let to_close: Vec<_> = positions
            .iter()
            .filter(|p| p.symbol == "BTCUSDT" && p.side.as_str() == "LONG")
            .map(|p| (p.position_id, CloseReason::RiskManager))
            .collect();

        assert_eq!(to_close.len(), 2, "Should close 2 BTCUSDT LONG positions");
    }

    // ═══════════════════════════════════════════════════════════
    // 13. CONFIG VALIDATION
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_config_default_valid() {
        let config = OrderManagerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_invalid_score_range() {
        let mut config = OrderManagerConfig::default();
        config.signal_score_min_1h = 0.9;
        config.signal_score_max_1h = 0.8;
        assert!(config.validate().is_err(), "min > max should fail validation");
    }

    #[test]
    fn test_config_invalid_leverage() {
        let mut config = OrderManagerConfig::default();
        config.leverage = 0;
        assert!(config.validate().is_err(), "leverage=0 should fail");

        config.leverage = 126;
        assert!(config.validate().is_err(), "leverage=126 should fail");
    }

    #[test]
    fn test_config_invalid_tf_percentages() {
        let mut config = OrderManagerConfig::default();
        config.tf_1h_pct = 50;
        config.tf_4h_pct = 30;
        config.tf_15m_pct = 10;
        // Sum = 90 ≠ 100
        assert!(config.validate().is_err(), "TF percentages must sum to 100");
    }

    // ═══════════════════════════════════════════════════════════
    // 14. EDGE CASES
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_pnl_zero_move() {
        let pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20);
        let (pnl, pnl_pct) = pos.calc_unrealized_pnl(50000.0);
        assert!((pnl - 0.0).abs() < 1e-10, "PnL should be 0 when price unchanged");
        assert!((pnl_pct - 0.0).abs() < 1e-10, "PnL% should be 0 when price unchanged");
    }

    #[test]
    fn test_tp_sl_not_hit_at_entry() {
        let pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20);
        assert!(!pos.is_tp_hit(50000.0), "TP should NOT hit at entry price");
        assert!(!pos.is_sl_hit(50000.0), "SL should NOT hit at entry price");
    }

    #[test]
    fn test_sl_tp_correct_for_long_direction() {
        // LONG: TP above entry, SL below entry
        let pos = make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20);
        assert!(pos.tp_price > pos.entry_price, "LONG TP should be above entry");
        assert!(pos.sl_price < pos.entry_price, "LONG SL should be below entry");
    }

    #[test]
    fn test_sl_tp_correct_for_short_direction() {
        // SHORT: TP below entry, SL above entry
        let pos = make_short_position(2, "ETHUSDT", 3000.0, 1.0, 10, 3100.0, 2800.0, 20);
        assert!(pos.tp_price < pos.entry_price, "SHORT TP should be below entry");
        assert!(pos.sl_price > pos.entry_price, "SHORT SL should be above entry");
    }

    #[test]
    fn test_multiple_positions_total_pnl() {
        let positions = vec![
            make_long_position(1, "BTCUSDT", 50000.0, 0.02, 10, 49000.0, 52000.0, 20),
            make_short_position(2, "ETHUSDT", 3000.0, 1.0, 10, 3100.0, 2800.0, 20),
            make_long_position(3, "SOLUSDT", 100.0, 10.0, 10, 95.0, 110.0, 20),
        ];

        // BTC: +20, ETH: +100, SOL: -20
        let current_prices = [51000.0, 2900.0, 98.0];
        let total_pnl: f64 = positions.iter().zip(current_prices.iter())
            .map(|(pos, &price)| pos.calc_unrealized_pnl(price).0)
            .sum();

        // BTC: (51000-50000)*0.02 = 20
        // ETH: -1*(2900-3000)*1.0 = 100
        // SOL: 1*(98-100)*10 = -20
        // Total = 20 + 100 - 20 = 100
        assert!((total_pnl - 100.0).abs() < 0.1, "Total PnL should be 100, got {}", total_pnl);
    }
}
