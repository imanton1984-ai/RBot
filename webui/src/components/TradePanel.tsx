import { useState } from 'react';
import { useTradingStore } from '../store';
import { apiService } from '../api';
import { TrendingUp, TrendingDown } from 'lucide-react';

export default function TradePanel() {
  const { currentPair, leverage, setLeverage, tradingMode, setTradingMode } = useTradingStore();
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [amount, setAmount] = useState('');
  const [price, setPrice] = useState('');
  const [reduceOnly, setReduceOnly] = useState(false);

  const handleOrder = async (side: 'long' | 'short') => {
    try {
      await apiService.placeOrder({
        pair: currentPair,
        side,
        type: orderType,
        price: orderType === 'limit' ? parseFloat(price) : undefined,
        amount_usdt: parseFloat(amount) || 0,
        leverage,
        reduce_only: reduceOnly,
      });
    } catch (error) {
      console.error('Failed to place order:', error);
    }
  };

  return (
    <div className="card h-full">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-textPrimary">Manual Trading</h3>
        <div className="flex items-center gap-2 text-xs">
          <span className={tradingMode === 'manual' ? 'text-binanceYellow' : 'text-textSecondary'}>
            Manual
          </span>
          <button
            className={`w-8 h-4 rounded-full relative transition-colors ${
              tradingMode === 'auto' ? 'bg-binanceYellow' : 'bg-panelAlt'
            }`}
            onClick={() => setTradingMode(tradingMode === 'manual' ? 'auto' : 'manual')}
          >
            <div
              className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${
                tradingMode === 'auto' ? 'translate-x-4' : 'translate-x-0.5'
              }`}
            />
          </button>
          <span className={tradingMode === 'auto' ? 'text-binanceYellow' : 'text-textSecondary'}>
            Auto
          </span>
        </div>
      </div>

      <div className="mb-4">
        <label className="text-xs text-textSecondary block mb-1">Pair</label>
        <div className="text-textPrimary font-medium">{currentPair}</div>
      </div>

      <div className="mb-4">
        <label className="text-xs text-textSecondary block mb-2">Leverage: {leverage}x</label>
        <input
          type="range"
          min="1"
          max="125"
          value={leverage}
          onChange={(e) => setLeverage(parseInt(e.target.value))}
          className="w-full accent-binanceYellow"
        />
        <div className="flex justify-between text-xs text-textSecondary mt-1">
          <span>1x</span>
          <span>25x</span>
          <span>50x</span>
          <span>75x</span>
          <span>125x</span>
        </div>
        <div className="flex gap-1 mt-2">
          {[5, 10, 20, 50].map((lev) => (
            <button
              key={lev}
              className={`flex-1 py-1 text-xs rounded ${
                leverage === lev
                  ? 'bg-binanceYellow text-black'
                  : 'bg-panelAlt text-textSecondary hover:bg-border'
              }`}
              onClick={() => setLeverage(lev)}
            >
              {lev}x
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-1 mb-4">
        <button
          className={`flex-1 py-1.5 text-xs rounded ${
            orderType === 'market'
              ? 'bg-binanceYellow text-black'
              : 'bg-panelAlt text-textSecondary'
          }`}
          onClick={() => setOrderType('market')}
        >
          Market
        </button>
        <button
          className={`flex-1 py-1.5 text-xs rounded ${
            orderType === 'limit'
              ? 'bg-binanceYellow text-black'
              : 'bg-panelAlt text-textSecondary'
          }`}
          onClick={() => setOrderType('limit')}
        >
          Limit
        </button>
      </div>

      <div className="mb-3">
        <label className="text-xs text-textSecondary block mb-1">Amount (USDT)</label>
        <div className="relative">
          <input
            type="number"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            placeholder="0.00"
            className="input-binance w-full"
          />
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-textSecondary">
            USDT
          </span>
        </div>
      </div>

      {orderType === 'limit' && (
        <div className="mb-3">
          <label className="text-xs text-textSecondary block mb-1">Price</label>
          <input
            type="number"
            value={price}
            onChange={(e) => setPrice(e.target.value)}
            placeholder="0.00"
            className="input-binance w-full"
          />
        </div>
      )}

      <label className="flex items-center gap-2 mb-4 cursor-pointer">
        <input
          type="checkbox"
          checked={reduceOnly}
          onChange={(e) => setReduceOnly(e.target.checked)}
          className="rounded border-border"
        />
        <span className="text-xs text-textSecondary">Reduce Only</span>
      </label>

      <div className="grid grid-cols-2 gap-2">
        <button
          className="btn-binance btn-binance-success py-3 flex items-center justify-center gap-2"
          onClick={() => handleOrder('long')}
        >
          <TrendingUp className="w-4 h-4" />
          Buy/Long
        </button>
        <button
          className="btn-binance btn-binance-danger py-3 flex items-center justify-center gap-2"
          onClick={() => handleOrder('short')}
        >
          <TrendingDown className="w-4 h-4" />
          Sell/Short
        </button>
      </div>

      <div className="mt-4 pt-4 border-t border-border text-xs">
        <div className="flex justify-between text-textSecondary mb-1">
          <span>Cost</span>
          <span>{amount || '0'} USDT</span>
        </div>
        <div className="flex justify-between text-textSecondary">
          <span>Max</span>
          <span>0.000 BTC</span>
        </div>
      </div>
    </div>
  );
}
