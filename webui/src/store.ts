import { create } from 'zustand';
import type { Balance, PnlOverview, Position, Alert, Strategy, Candle, Indicator, Signal, ConnectionStatus, TradingOptions, AutoTradingState } from './types';
import type { TerminalLog } from './components/TerminalWidget';

interface WsState {
  connected: boolean;
  lastPrice: number;
  priceChange24h: number;
  priceChange1h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  setConnected: (v: boolean) => void;
  setLastPrice: (v: number) => void;
  setPriceChange24h: (v: number) => void;
  setPriceChange1h: (v: number) => void;
  setVolume24h: (v: number) => void;
  setHigh24h: (v: number) => void;
  setLow24h: (v: number) => void;
}

interface TradingState {
  currentPair: string;
  currentTf: number;
  leverage: number;
  tradingMode: 'manual' | 'auto';
  autoTrading: AutoTradingState;
  setCurrentPair: (pair: string) => void;
  setCurrentTf: (tf: number) => void;
  setLeverage: (lev: number) => void;
  setTradingMode: (mode: 'manual' | 'auto') => void;
  setAutoTrading: (state: AutoTradingState) => void;
}

interface UiState {
  positionsView: 'open' | 'history';
  chartView: 'chart' | 'positions';
  tradingOptionsModalOpen: boolean;
  orderOptionsModalOpen: boolean;
  signalsModalOpen: boolean;
  setPositionsView: (view: 'open' | 'history') => void;
  setChartView: (view: 'chart' | 'positions') => void;
  setTradingOptionsModalOpen: (open: boolean) => void;
  setOrderOptionsModalOpen: (open: boolean) => void;
  setSignalsModalOpen: (open: boolean) => void;
}

interface DataState {
  candles: Candle[];
  indicators: Record<string, Indicator[]>;
  positions: Position[];
  balance: Balance | null;
  pnlOverview: PnlOverview | null;
  alerts: Alert[];
  strategies: Strategy[];
  signals: Signal[];
  connections: ConnectionStatus | null;
  tradingOptions: TradingOptions | null;
  terminalLogs: TerminalLog[];
  setCandles: (candles: Candle[]) => void;
  setIndicators: (type: string, data: Indicator[]) => void;
  setPositions: (positions: Position[]) => void;
  setBalance: (balance: Balance) => void;
  setPnlOverview: (pnl: PnlOverview) => void;
  setAlerts: (alerts: Alert[]) => void;
  setStrategies: (strategies: Strategy[]) => void;
  setSignals: (signals: Signal[]) => void;
  setConnections: (connections: ConnectionStatus) => void;
  setTradingOptions: (options: TradingOptions) => void;
  addTerminalLog: (level: 'error' | 'warn' | 'info', message: string) => void;
  clearTerminalLogs: () => void;
  updateCandle: (candle: Candle) => void;
  updatePosition: (position: Position) => void;
}

export const useTradingStore = create<TradingState>((set) => ({
  currentPair: 'BTCUSDT',
  currentTf: 60,
  leverage: 10,
  tradingMode: 'manual',
  autoTrading: { is_running: false, trading_mode: 'off' },
  setCurrentPair: (pair) => set({ currentPair: pair }),
  setCurrentTf: (tf) => set({ currentTf: tf }),
  setLeverage: (lev) => set({ leverage: lev }),
  setTradingMode: (mode) => set({ tradingMode: mode }),
  setAutoTrading: (state) => set({ autoTrading: state }),
}));

export const useUiStore = create<UiState>((set) => ({
  positionsView: 'open',
  chartView: 'chart',
  tradingOptionsModalOpen: false,
  orderOptionsModalOpen: false,
  signalsModalOpen: false,
  setPositionsView: (view) => set({ positionsView: view }),
  setChartView: (view) => set({ chartView: view }),
  setTradingOptionsModalOpen: (open) => set({ tradingOptionsModalOpen: open }),
  setOrderOptionsModalOpen: (open) => set({ orderOptionsModalOpen: open }),
  setSignalsModalOpen: (open) => set({ signalsModalOpen: open }),
}));

export const useDataStore = create<DataState>((set) => ({
  candles: [],
  indicators: {},
  positions: [],
  balance: null,
  pnlOverview: null,
  alerts: [],
  strategies: [],
  signals: [],
  connections: null,
  tradingOptions: null,
  terminalLogs: [],
  setCandles: (candles) => set({ candles }),
  setIndicators: (type, data) => set((state) => ({ indicators: { ...state.indicators, [type]: data } })),
  setPositions: (positions) => set({ positions }),
  setBalance: (balance) => set({ balance }),
  setPnlOverview: (pnlOverview) => set({ pnlOverview }),
  setAlerts: (alerts) => set({ alerts }),
  setStrategies: (strategies) => set({ strategies }),
  setSignals: (signals) => set({ signals }),
  setConnections: (connections) => set({ connections }),
  setTradingOptions: (tradingOptions) => set({ tradingOptions }),
  addTerminalLog: (level, message) => set((state) => {
    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false });
    const log: TerminalLog = { id: Date.now() + Math.random(), time, level, message };
    return { terminalLogs: [...state.terminalLogs.slice(-99), log] }; // Keep max 100
  }),
  clearTerminalLogs: () => set({ terminalLogs: [] }),
  updateCandle: (candle) => set((state) => ({ candles: state.candles.length > 0 ? [...state.candles.slice(0, -1), candle] : [candle] })),
  updatePosition: (position) => set((state) => ({ positions: state.positions.map(p => p.id === position.id ? position : p) })),
}));

export const useWsStore = create<WsState>((set) => ({
  connected: false,
  lastPrice: 0,
  priceChange24h: 0,
  priceChange1h: 0,
  volume24h: 0,
  high24h: 0,
  low24h: 0,
  setConnected: (v) => set({ connected: v }),
  setLastPrice: (v) => set({ lastPrice: v }),
  setPriceChange24h: (v) => set({ priceChange24h: v }),
  setPriceChange1h: (v) => set({ priceChange1h: v }),
  setVolume24h: (v) => set({ volume24h: v }),
  setHigh24h: (v) => set({ high24h: v }),
  setLow24h: (v) => set({ low24h: v }),
}));
