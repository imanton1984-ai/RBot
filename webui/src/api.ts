import axios from 'axios';
import type {
  Candle, Indicator, Pair, MarketSummary, Position, Signal, Balance,
  PnlOverview, Alert, ConnectionStatus, TradingOptions,
  OrderOptions, ManualOrderRequest, AutoTradingState,
  FullStatistics, AccountBalances, TransferRequest, TransferResponse,
} from './types';

const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
});

export const apiService = {
  // ─── Pairs ──────────────────────────────────────────────────
  getPairs: async (search?: string, limit?: number): Promise<Pair[]> => {
    const params = new URLSearchParams();
    if (search) params.append('search', search);
    if (limit) params.append('limit', limit.toString());
    const { data } = await api.get<Pair[]>('/pairs', { params });
    return data;
  },

  // ─── Market ─────────────────────────────────────────────────
  getMarketSummary: async (pair: string, tf: number): Promise<MarketSummary> => {
    const { data } = await api.get<MarketSummary>('/market/summary', {
      params: { pair, tf },
    });
    return data;
  },

  // ─── Candles ────────────────────────────────────────────────
  getCandles: async (pair: string, tf: number, limit?: number): Promise<Candle[]> => {
    const { data } = await api.get<Candle[]>('/candles', {
      params: { pair, tf, limit },
    });
    return data;
  },

  // ─── Indicators ──────────────────────────────────────────────
  getIndicators: async (pair: string, tf: number, type: string, limit?: number): Promise<Indicator[]> => {
    const { data } = await api.get<Indicator[]>('/indicators', {
      params: { pair, tf, type, limit },
    });
    return data;
  },

  // ─── Signals ────────────────────────────────────────────────
  getSignals: async (pair?: string, tf?: number, limit?: number): Promise<Signal[]> => {
    const params = new URLSearchParams();
    if (pair) params.append('pair', pair);
    if (tf) params.append('tf', tf.toString());
    if (limit) params.append('limit', limit.toString());
    const { data } = await api.get<Signal[]>('/signals', { params });
    return data;
  },

  // ─── Positions ──────────────────────────────────────────────
  getOpenPositions: async (): Promise<Position[]> => {
    const { data } = await api.get<Position[]>('/positions/open');
    return data;
  },

  getPositionsHistory: async (
    from?: number,
    to?: number,
    pair?: string,
    closeType?: string
  ): Promise<any[]> => {
    const params = new URLSearchParams();
    if (from) params.append('from', from.toString());
    if (to) params.append('to', to.toString());
    if (pair) params.append('pair', pair);
    if (closeType) params.append('close_type', closeType);
    const { data } = await api.get<any[]>('/positions/history', { params });
    return data;
  },

  // ─── Statistics ───────────────────────────────────────────
  getStatistics: async (range?: string, interval?: string): Promise<FullStatistics> => {
    const params = new URLSearchParams();
    if (range) params.append('range', range);
    if (interval) params.append('interval', interval);
    const { data } = await api.get<FullStatistics>('/statistics', { params });
    return data;
  },

  // ─── PnL & Balance ─────────────────────────────────────────
  getPnlOverview: async (range?: string): Promise<PnlOverview> => {
    const { data } = await api.get<PnlOverview>('/pnl/overview', {
      params: { range },
    });
    return data;
  },

  getBalance: async (): Promise<Balance> => {
    const { data } = await api.get<Balance>('/balance');
    return data;
  },

  // ─── Connections ────────────────────────────────────────────
  getConnections: async (): Promise<ConnectionStatus> => {
    const { data } = await api.get<ConnectionStatus>('/connections');
    return data;
  },

  // ─── Alerts (from risk_manager) ────────────────────────────
  getAlerts: async (limit?: number): Promise<Alert[]> => {
    const { data } = await api.get<Alert[]>('/alerts', {
      params: { limit },
    });
    return data;
  },

  // ─── Trading Options (order_settings.toml) ─────────────────
  getTradingOptions: async (): Promise<TradingOptions> => {
    const { data } = await api.get<TradingOptions>('/trading-options');
    return data;
  },

  saveTradingOptions: async (options: TradingOptions): Promise<void> => {
    await api.post('/trading-options', options);
  },

  // ─── Order Options (order_manager.toml + risk_manager.toml) ─
  getOrderOptions: async (): Promise<OrderOptions> => {
    const { data } = await api.get<OrderOptions>('/order-options');
    return data;
  },

  saveOrderOptions: async (options: OrderOptions): Promise<void> => {
    await api.post('/order-options', options);
  },

  // ─── Trading ────────────────────────────────────────────────
  placeOrder: async (order: ManualOrderRequest): Promise<void> => {
    await api.post('/trade/order', order);
  },

  closePosition: async (positionId: number): Promise<void> => {
    await api.post('/trade/close', { position_id: positionId });
  },

  // ─── Candles Left Update ────────────────────────────────────
  updateCandlesLeft: async (positionId: number, candlesLeft: number): Promise<void> => {
    await api.post('/trade/update-candles-left', { position_id: positionId, candles_left: candlesLeft });
  },

  // ─── Control ────────────────────────────────────────────────
  emergencyStop: async (): Promise<void> => {
    await api.post('/control/emergency_stop');
  },

  startTrading: async (): Promise<AutoTradingState> => {
    const { data } = await api.post<AutoTradingState>('/control/start_trading');
    return data;
  },

  stopTrading: async (): Promise<AutoTradingState> => {
    const { data } = await api.post<AutoTradingState>('/control/stop_trading');
    return data;
  },

  getAutoTradingState: async (): Promise<AutoTradingState> => {
    const { data } = await api.get<AutoTradingState>('/control/trading_state');
    return data;
  },

  // ─── Account (Spot/Futures) ──────────────────────────────
  getAccountBalances: async (): Promise<AccountBalances> => {
    const { data } = await api.get<AccountBalances>('/account/balances');
    return data;
  },

  transferBetweenAccounts: async (request: TransferRequest): Promise<TransferResponse> => {
    const { data } = await api.post<TransferResponse>('/account/transfer', request);
    return data;
  },
};
