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
}

export interface PnlOverview {
  closed_pnl: number;
  unrealized_pnl: number;
  win_rate: number;
  today_trades: number;
  equity_points: PnlPoint[];
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
}

export interface Strategy {
  id: string;
  name: string;
  enabled: boolean;
  priority: number;
  description: string;
}

export interface WebUiSettings {
  leverage_default: number;
  max_open_orders: number;
  max_risk_pct: number;
  order_timeout_bars: number;
  ws_update_rate_ms: number;
}
