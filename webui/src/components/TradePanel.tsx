import { useState, useEffect } from 'react';
import { useTradingStore, useDataStore } from '../store';
import { apiService } from '../api';
import { useQuery } from '@tanstack/react-query';
import { TrendingUp, TrendingDown, Settings2 } from 'lucide-react';

export default function TradePanel() {
  const { currentPair, leverage, setLeverage, tradingMode, setTradingMode } = useTradingStore();
  const positions = useDataStore((s) => s.positions);
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [amount, setAmount] = useState('');
  const [price, setPrice] = useState('');
  const [takeProfit, setTakeProfit] = useState('');
  const [stopLoss, setStopLoss] = useState('');
  const [entryPrice, setEntryPrice] = useState('');
  const [reduceOnly, setReduceOnly] = useState(false);

  // Load trading options for auto mode display
  const { data: tradingOptions } = useQuery({
    queryKey: ['trading-options'],
    queryFn: () => apiService.getTradingOptions(),
    refetchInterval: 30000,
  });

  const handleOrder = async (side: 'long' | 'short') => {
    try {
      await apiService.placeOrder({
        pair: currentPair,
        side,
        type: orderType,
        price: orderType === 'limit' ? parseFloat(price) : undefined,
        amount_usdt: parseFloat(amount) || 0,
        leverage,
        take_profit: parseFloat(takeProfit) || 0,
        stop_loss: parseFloat(stopLoss) || 0,
        entry_price: entryPrice ? parseFloat(entryPrice) : undefined,
        reduce_only: reduceOnly,
      });
    } catch (error) {
      console.error('Failed to place order:', error);
    }
  };

  // Auto mode — display current settings
  if (tradingMode === 'auto') {
    const opts = tradingOptions;
    return (
      <div className="card h-full">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-textPrimary">Auto Trading</h3>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-textSecondary">Manual</span>
            <button
              className="w-8 h-4 rounded-full relative transition-colors bg-binanceYellow"
              onClick={() => setTradingMode('manual')}
            >
              <div className="absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform translate-x-4" />
            </button>
            <span className="text-binanceYellow">Auto</span>
          </div>
        </div>

        <div className="space-y-4">
          {/* Current Auto Settings Summary */}
          <div className="p-3 rounded-lg bg-panelAlt border border-border">
            <div className="flex items-center gap-2 mb-3">
              <Settings2 className="w-4 h-4 text-binanceYellow" />
              <span className="text-xs font-medium text-textPrimary">Current Settings</span>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-textSecondary">Strategy</span>
                <span className="text-binanceYellow font-medium">
                  {opts?.strategy_type === 'ml_super_entry' ? 'ML Super Entry' : opts?.strategy_type || '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-textSecondary">Leverage</span>
                <span className="text-textPrimary">{opts?.leverage || leverage}x</span>
              </div>
              <div className="flex justify-between">
                <span className="text-textSecondary">Order Size</span>
                <span className="text-textPrimary">
                  {opts?.trade_size_type === 'fixed_usdt'
                    ? `${opts?.trade_size_value} USDT`
                    : `${opts?.trade_size_value}%`}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-textSecondary">Order Type</span>
                <span className="text-textPrimary">
                  {opts?.order_type === 'futures_oco' ? 'Futures OCO' : opts?.order_type || '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-textSecondary">Max Orders</span>
                <span className="text-textPrimary">{opts?.max_orders_at_a_time || 10}</span>
              </div>
            </div>
          </div>

          {/* Open Positions Count */}
          <div className="p-3 rounded-lg bg-panelAlt border border-border">
            <div className="flex justify-between text-xs">
              <span className="text-textSecondary">Open Positions</span>
              <span className="text-textPrimary font-medium">{positions.length}</span>
            </div>
          </div>

          {/* Trading Mode */}
          <div className="p-3 rounded-lg bg-panelAlt border border-border">
            <div className="flex justify-between items-center text-xs">
              <span className="text-textSecondary">Trading Mode</span>
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${opts?.trading_mode === 'auto' ? 'bg-bull/20 text-bull' :
                opts?.trading_mode === 'manual' ? 'bg-binanceYellow/20 text-binanceYellow' :
                  'bg-bear/20 text-bear'
                }`}>
                {opts?.trading_mode?.toUpperCase() || 'OFF'}
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Manual mode — full trading panel
  return (
    <div className="card h-full overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-textPrimary">Manual Trading</h3>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-binanceYellow">Manual</span>
          <button
            className="w-8 h-4 rounded-full relative transition-colors bg-panelAlt"
            onClick={() => setTradingMode('auto')}
          >
            <div className="absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform translate-x-0.5" />
          </button>
          <span className="text-textSecondary">Auto</span>
        </div>
      </div>

      <div className="mb-3">
        <label className="text-xs text-textSecondary block mb-1">Pair</label>
        <div className="text-textPrimary font-medium">{currentPair}</div>
      </div>

      <div className="mb-3">
        <label className="text-xs text-textSecondary block mb-2">Leverage: {leverage}x</label>
        <input
          type="range" min="1" max="125" value={leverage}
          onChange={(e) => setLeverage(parseInt(e.target.value))}
          className="w-full accent-binanceYellow"
        />
        <div className="flex gap-1 mt-1">
          {[5, 10, 20, 50].map((lev) => (
            <button key={lev}
              className={`flex-1 py-1 text-xs rounded ${leverage === lev ? 'bg-binanceYellow text-black' : 'bg-panelAlt text-textSecondary hover:bg-border'}`}
              onClick={() => setLeverage(lev)}
            >{lev}x</button>
          ))}
        </div>
      </div>

      <div className="flex gap-1 mb-3">
        <button
          className={`flex-1 py-1.5 text-xs rounded ${orderType === 'market' ? 'bg-binanceYellow text-black' : 'bg-panelAlt text-textSecondary'}`}
          onClick={() => setOrderType('market')}
        >Market</button>
        <button
          className={`flex-1 py-1.5 text-xs rounded ${orderType === 'limit' ? 'bg-binanceYellow text-black' : 'bg-panelAlt text-textSecondary'}`}
          onClick={() => setOrderType('limit')}
        >Limit</button>
      </div>

      <div className="mb-3">
        <label className="text-xs text-textSecondary block mb-1">Amount (USDT)</label>
        <div className="relative">
          <input type="number" value={amount} onChange={(e) => setAmount(e.target.value)}
            placeholder="0.00" className="input-binance w-full" />
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-textSecondary">USDT</span>
        </div>
      </div>

      {/* Entry Price (for limit orders) */}
      {orderType === 'limit' && (
        <div className="mb-3">
          <label className="text-xs text-textSecondary block mb-1">Entry Price</label>
          <input type="number" value={entryPrice} onChange={(e) => setEntryPrice(e.target.value)}
            placeholder="0.00" className="input-binance w-full" />
        </div>
      )}

      {/* Take Profit */}
      <div className="mb-3">
        <label className="text-xs text-bull block mb-1">Take Profit</label>
        <input type="number" value={takeProfit} onChange={(e) => setTakeProfit(e.target.value)}
          placeholder="0.00" className="input-binance w-full border-bull/30 focus:border-bull" />
      </div>

      {/* Stop Loss */}
      <div className="mb-3">
        <label className="text-xs text-bear block mb-1">Stop Loss</label>
        <input type="number" value={stopLoss} onChange={(e) => setStopLoss(e.target.value)}
          placeholder="0.00" className="input-binance w-full border-bear/30 focus:border-bear" />
      </div>

      <label className="flex items-center gap-2 mb-4 cursor-pointer">
        <input type="checkbox" checked={reduceOnly}
          onChange={(e) => setReduceOnly(e.target.checked)}
          className="rounded border-border" />
        <span className="text-xs text-textSecondary">Reduce Only</span>
      </label>

      <div className="grid grid-cols-2 gap-2">
        <button
          className="btn-binance btn-binance-success py-3 flex items-center justify-center gap-2"
          onClick={() => handleOrder('long')}
        >
          <TrendingUp className="w-4 h-4" /> Buy/Long
        </button>
        <button
          className="btn-binance btn-binance-danger py-3 flex items-center justify-center gap-2"
          onClick={() => handleOrder('short')}
        >
          <TrendingDown className="w-4 h-4" /> Sell/Short
        </button>
      </div>

      <div className="mt-3 pt-3 border-t border-border text-xs">
        <div className="flex justify-between text-textSecondary mb-1">
          <span>Cost</span>
          <span>{amount || '0'} USDT</span>
        </div>
        <div className="flex justify-between text-textSecondary">
          <span>Notional</span>
          <span>{((parseFloat(amount) || 0) * leverage).toFixed(2)} USDT</span>
        </div>
      </div>
    </div>
  );
}
