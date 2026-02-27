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
  tf_minutes: number;
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
  signal_score_min_1m: number;
  signal_score_max_1m: number;
  signal_score_min_5m: number;
  signal_score_max_5m: number;
  signal_score_min_15m: number;
  signal_score_max_15m: number;
  signal_score_min_1h: number;
  signal_score_max_1h: number;
  signal_score_min_4h: number;
  signal_score_max_4h: number;
  signal_score_min_1d: number;
  signal_score_max_1d: number;
  max_hold_bars: number;
  tf_1m_pct: number;
  tf_5m_pct: number;
  tf_15m_pct: number;
  tf_1h_pct: number;
  tf_4h_pct: number;
  tf_1d_pct: number;
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

// ─── Statistics Types ──────────────────────────────────────
export interface StatsSummary {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_pnl: number;
  total_pnl_pct: number;
  avg_win: number;
  avg_win_pct: number;
  avg_loss: number;
  avg_loss_pct: number;
  profit_factor: number;
  best_trade: number;
  best_trade_pct: number;
  worst_trade: number;
  worst_trade_pct: number;
  avg_trade_duration_hours: number;
  max_consecutive_wins: number;
  max_consecutive_losses: number;
  expectancy: number;
  sharpe_approx: number;
}

export interface DayPartStats {
  bucket: string;
  hours: string;
  trades: number;
  wins: number;
  losses: number;
  pnl: number;
  pnl_pct: number;
  win_rate: number;
  avg_pnl: number;
  best_trade: number;
  worst_trade: number;
}

export interface PnlTimePoint {
  t: number;
  cumulative_pnl: number;
  balance: number;
}

export interface PnlTimeline {
  points: PnlTimePoint[];
  start_date: string;
  end_date: string;
}

export interface TradeDistribution {
  by_pair: PairStats[];
  by_timeframe: TfStats[];
  by_close_type: CloseTypeStats[];
  by_side: SideStats[];
}

export interface PairStats {
  pair: string;
  trades: number;
  wins: number;
  pnl: number;
  win_rate: number;
}

export interface TfStats {
  tf: number;
  trades: number;
  wins: number;
  pnl: number;
  win_rate: number;
}

export interface CloseTypeStats {
  close_type: string;
  count: number;
  pnl: number;
  win_rate: number;
}

export interface SideStats {
  side: string;
  trades: number;
  wins: number;
  pnl: number;
  win_rate: number;
}

export interface BestWorstTrades {
  best: HistoryPosition[];
  worst: HistoryPosition[];
}

export interface TimeSlotStats {
  label: string;
  trades: number;
  wins: number;
  losses: number;
  pnl: number;
  win_rate: number;
  avg_pnl: number;
}

export interface TimeBasedStats {
  period: string;
  data: TimeSlotStats[];
}

export interface MonthlyStats {
  month: string;
  trades: number;
  wins: number;
  pnl: number;
  win_rate: number;
}

export interface FullStatistics {
  summary: StatsSummary;
  timeline: PnlTimeline;
  distribution: TradeDistribution;
  best_worst: BestWorstTrades;
  hourly: TimeBasedStats;
  daily: TimeBasedStats;
  weekly: TimeBasedStats;
  monthly: MonthlyStats[];
  day_parts: DayPartStats[];
}
