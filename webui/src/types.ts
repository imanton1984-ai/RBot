export interface Candle {
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

export interface Indicator {
  t: number;
  value?: number;
  macd?: number;
  signal?: number;
  hist?: number;
}

export interface Pair {
  symbol: string;
  symbol_id: number;
}

export interface MarketSummary {
  price: number;
  volume_24h: number;
  change_24h: number;
  change_1h: number;
  high_24h: number;
  low_24h: number;
}

export interface Position {
  id: number;
  pair: string;
  side: 'LONG' | 'SHORT';
  qty: number;
  entry_price: number;
  current_price: number;
  stop_loss: number;
  take_profit: number;
  pnl_usdt: number;
  pnl_pct: number;
  status: string;
  open_time: string;
  candles_left: number;
  leverage: number;
}

export interface HistoryPosition {
  id: number;
  pair: string;
  side: string;
  qty: number;
  entry_price: number;
  close_price: number;
  close_type: string;
  pnl_usdt: number;
  pnl_pct: number;
  open_time: string;
  close_time: string;
}

export interface Signal {
  id: string;
  time: string;
  pair: string;
  tf: number;
  side: number;
  score: number;
  entry_price: number;
  sl_price: number;
  tp_price: number;
  strategy: string;
  status: string;
  p_super: number;
}

export interface Balance {
  overall: number;
  in_orders: number;
  available: number;
  wallet_balance: number;
  unrealized_pnl: number;
}

export interface PnlOverview {
  closed_pnl: number;
  unrealized_pnl: number;
  win_rate: number;
  today_trades: number;
  equity_points: PnlPoint[];
  overall_balance: number;
}

export interface PnlPoint {
  t: number;
  value: number;
}

export interface Alert {
  id: number;
  pair: string;
  alert_type: string;
  time_ago: string;
  message: string;
  severity: string;
  source: string;
  timestamp: string;
}

export interface Strategy {
  id: string;
  name: string;
  enabled: boolean;
  priority: number;
  description: string;
}

// ─── Connection Status ──────────────────────────────────────
export interface ConnectionStatus {
  database: boolean;
  redpanda: boolean;
  rest_api: boolean;
  websocket: boolean;
  account: boolean;
}

// ─── Trading Options (order_settings.toml) ──────────────────
export interface TradingOptions {
  leverage: number;
  max_orders_at_a_time: number;
  trade_size_type: 'fixed_usdt' | 'percent_depo';
  trade_size_value: number;
  strategy_type: 'ml_super_entry' | 'level_strategy';
  order_type: 'futures_oco';
  trading_mode: 'auto' | 'manual' | 'off';
}

// ─── Order Manager Options (order_manager.toml) ────────────
export interface OrderManagerOptions {
  signal_score_min: number;
  signal_score_max: number;
  max_hold_bars: number;
  tf_1h_pct: number;
  tf_4h_pct: number;
  tf_15m_pct: number;
}

// ─── Risk Manager Options (subset of risk_manager.toml) ────
export interface RiskManagerOptions {
  btc_alert_threshold_pct: number;
  alt_alert_threshold_pct: number;
  volume_spike_threshold: number;
}

// ─── Combined Order Options ─────────────────────────────────
export interface OrderOptions {
  order_manager: OrderManagerOptions;
  risk_manager: RiskManagerOptions;
}

// ─── Manual Order Request ───────────────────────────────────
export interface ManualOrderRequest {
  pair: string;
  side: 'long' | 'short';
  type: 'market' | 'limit';
  price?: number;
  amount_usdt: number;
  leverage: number;
  take_profit: number;
  stop_loss: number;
  entry_price?: number;
  reduce_only: boolean;
}

// ─── Auto Trading State ─────────────────────────────────────
export interface AutoTradingState {
  is_running: boolean;
  trading_mode: 'auto' | 'manual' | 'off';
}

// Legacy compat
export interface WebUiSettings {
  leverage_default: number;
  max_open_orders: number;
  max_risk_pct: number;
  order_timeout_bars: number;
  ws_update_rate_ms: number;
}
