import axios from 'axios';
import type { Candle, Indicator, Pair, MarketSummary, Position, Signal, Balance, PnlOverview, Alert, Strategy, WebUiSettings } from './types';

const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
});

export const apiService = {
  // Pairs
  getPairs: async (search?: string, limit?: number): Promise<Pair[]> => {
    const params = new URLSearchParams();
    if (search) params.append('search', search);
    if (limit) params.append('limit', limit.toString());
    const { data } = await api.get<Pair[]>('/pairs', { params });
    return data;
  },

  // Market
  getMarketSummary: async (pair: string, tf: number): Promise<MarketSummary> => {
    const { data } = await api.get<MarketSummary>('/market/summary', {
      params: { pair, tf },
    });
    return data;
  },

  // Candles
  getCandles: async (pair: string, tf: number, limit?: number): Promise<Candle[]> => {
    const { data } = await api.get<Candle[]>('/candles', {
      params: { pair, tf, limit },
    });
    return data;
  },

  // Indicators
  getIndicators: async (pair: string, tf: number, type: string, limit?: number): Promise<Indicator[]> => {
    const { data } = await api.get<Indicator[]>('/indicators', {
      params: { pair, tf, type, limit },
    });
    return data;
  },

  // Signals
  getSignals: async (pair?: string, tf?: number, limit?: number): Promise<Signal[]> => {
    const params = new URLSearchParams();
    if (pair) params.append('pair', pair);
    if (tf) params.append('tf', tf.toString());
    if (limit) params.append('limit', limit.toString());
    const { data } = await api.get<Signal[]>('/signals', { params });
    return data;
  },

  // Positions
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

  // PnL & Balance
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

  // Alerts
  getAlerts: async (limit?: number): Promise<Alert[]> => {
    const { data } = await api.get<Alert[]>('/alerts', {
      params: { limit },
    });
    return data;
  },

  // Settings
  getOptions: async (): Promise<WebUiSettings> => {
    const { data } = await api.get<WebUiSettings>('/options');
    return data;
  },

  saveOptions: async (settings: WebUiSettings): Promise<void> => {
    await api.post('/options', settings);
  },

  // Strategies
  getStrategies: async (): Promise<Strategy[]> => {
    const { data } = await api.get<Strategy[]>('/strategies');
    return data;
  },

  toggleStrategy: async (id: string, enabled: boolean): Promise<void> => {
    await api.post('/strategy/toggle', { id, enabled });
  },

  // Trading
  placeOrder: async (order: {
    pair: string;
    side: string;
    type: string;
    price?: number;
    amount_usdt: number;
    leverage: number;
    reduce_only: boolean;
  }): Promise<void> => {
    await api.post('/trade/order', order);
  },

  closePosition: async (positionId: number): Promise<void> => {
    await api.post('/trade/close', { position_id: positionId });
  },

  // Control
  emergencyStop: async (): Promise<void> => {
    await api.post('/control/emergency_stop');
  },

  reloadBase: async (): Promise<void> => {
    await api.post('/control/reload_base');
  },

  startTrading: async (): Promise<void> => {
    await api.post('/control/start_trading');
  },
};
