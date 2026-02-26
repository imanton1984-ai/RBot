import { useEffect } from 'react';
import { useUiStore, useDataStore, useTradingStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { X } from 'lucide-react';
import { formatPrice, formatQty, formatTf } from '../utils/format';

export default function PositionsPanel() {
  const { positionsView, setPositionsView, setChartView } = useUiStore();
  const { setCurrentPair, setCurrentTf } = useTradingStore();
  const positions = useDataStore((s) => s.positions);
  const setPositions = useDataStore((s) => s.setPositions);

  // Open positions — refresh every 2s for near-real-time PnL updates
  // Position tracker updates DB every 3s, so 2s polling gives smooth updates
  const { data: openPositions } = useQuery({
    queryKey: ['positions-open'],
    queryFn: () => apiService.getOpenPositions(),
    refetchInterval: 2000,
    staleTime: 1500,
    enabled: positionsView === 'open',
    retry: 1,
    // Refetch on window focus to ensure fresh data when switching back
    refetchOnWindowFocus: true,
  });

  // History — from DB, refresh less frequently
  const { data: historyPositions } = useQuery({
    queryKey: ['positions-history'],
    queryFn: () => apiService.getPositionsHistory(),
    refetchInterval: 15000,
    staleTime: 10000,
    enabled: positionsView === 'history',
    retry: 1,
  });

  useEffect(() => {
    if (positionsView === 'open' && openPositions) {
      setPositions(openPositions);
    }
  }, [openPositions, positionsView, setPositions]);

  const currentPositions = positionsView === 'open' ? (openPositions || positions) : (historyPositions || []);

  const handleClosePosition = async (positionId: number) => {
    if (confirm('Close this position?')) {
      try {
        await apiService.closePosition(positionId);
      } catch (error) {
        console.error('Failed to close position:', error);
      }
    }
  };

  const handleOpenInChart = (pos: any) => {
    if (!pos?.pair) return;

    setCurrentPair(pos.pair);

    // For open positions, use tf_minutes
    if (pos.tf_minutes) {
      setCurrentTf(Number(pos.tf_minutes));
    }

    setChartView('chart');
  };

  return (
    <div className="h-full flex flex-col">
      <div className="h-10 border-b border-border flex items-center px-4 gap-2 shrink-0">
        <button
          className={`px-3 py-1.5 rounded text-sm font-medium ${positionsView === 'open'
            ? 'bg-binanceYellow text-black'
            : 'text-textSecondary hover:bg-panelAlt'
            }`}
          onClick={() => setPositionsView('open')}
        >
          Open Positions
          {positionsView === 'open' && openPositions && openPositions.length > 0 && (
            <span className="ml-1.5 px-1.5 py-0.5 rounded bg-black/20 text-xs">{openPositions.length}</span>
          )}
        </button>
        <button
          className={`px-3 py-1.5 rounded text-sm font-medium ${positionsView === 'history'
            ? 'bg-binanceYellow text-black'
            : 'text-textSecondary hover:bg-panelAlt'
            }`}
          onClick={() => setPositionsView('history')}
        >
          History
        </button>
      </div>

      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-panel text-textSecondary">
            {positionsView === 'open' ? (
              <tr>
                <th className="text-left font-normal px-4 py-2">Pair</th>
                <th className="text-left font-normal px-4 py-2">Side</th>
                <th className="text-left font-normal px-4 py-2">Qty</th>
                <th className="text-left font-normal px-4 py-2">Entry Price</th>
                <th className="text-left font-normal px-4 py-2">Current Price</th>
                <th className="text-left font-normal px-4 py-2">Stop Loss</th>
                <th className="text-left font-normal px-4 py-2">Take Profit</th>
                <th className="text-left font-normal px-4 py-2">PnL</th>
                <th className="text-left font-normal px-4 py-2">Bars Left</th>
                <th className="text-left font-normal px-4 py-2">Action</th>
                <th className="text-left font-normal px-4 py-2">TF</th>
              </tr>
            ) : (
              <tr>
                <th className="text-left font-normal px-4 py-2">Pair</th>
                <th className="text-left font-normal px-4 py-2">Side</th>
                <th className="text-left font-normal px-4 py-2">Qty</th>
                <th className="text-left font-normal px-4 py-2">Entry</th>
                <th className="text-left font-normal px-4 py-2">Close</th>
                <th className="text-left font-normal px-4 py-2">Type</th>
                <th className="text-left font-normal px-4 py-2">PnL</th>
                <th className="text-left font-normal px-4 py-2">Close Time</th>
              </tr>
            )}
          </thead>
          <tbody>
            {currentPositions.length > 0 ? (
              currentPositions.map((pos: any, idx: number) => (
                <tr
                  key={pos.id || idx}
                  className="border-t border-border hover:bg-panelAlt transition-colors"
                  onDoubleClick={() => positionsView === 'open' && handleOpenInChart(pos)}
                  title={positionsView === 'open' ? 'Double-click to open chart' : undefined}
                >
                  {positionsView === 'open' ? (
                    <>
                      <td className="px-4 py-2 text-textPrimary font-medium">{pos.pair}</td>
                      <td className={`px-4 py-2 ${pos.side === 'LONG' ? 'text-bull' : 'text-bear'}`}>
                        {pos.side}
                      </td>
                      <td className="px-4 py-2 text-textPrimary">{formatQty(pos.qty)}</td>
                      <td className="px-4 py-2 text-textPrimary">${formatPrice(pos.entry_price)}</td>
                      <td className="px-4 py-2 text-textPrimary">${formatPrice(pos.current_price)}</td>
                      <td className="px-4 py-2 text-bear">
                        {pos.stop_loss && pos.stop_loss > 0 ? `$${formatPrice(pos.stop_loss)}` : '-'}
                      </td>
                      <td className="px-4 py-2 text-bull">
                        {pos.take_profit && pos.take_profit > 0 ? `$${formatPrice(pos.take_profit)}` : '-'}
                      </td>
                      <td className={`px-4 py-2 ${pos.pnl_usdt >= 0 ? 'text-bull' : 'text-bear'}`}>
                        {pos.pnl_usdt >= 0 ? '+' : ''}${pos.pnl_usdt?.toFixed(2)}
                      </td>
                      <td className="px-4 py-2 text-binanceYellow font-medium">
                        {pos.candles_left ?? 0}
                      </td>
                      <td className="px-4 py-2">
                        <button
                          className="text-bear hover:text-bear/80 transition-colors"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleClosePosition(pos.id);
                          }}
                          onDoubleClick={(e) => e.stopPropagation()}
                          title="Close position"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </td>
                      <td className="px-4 py-2 text-textSecondary font-medium">
                        {formatTf(pos.tf_minutes)}
                      </td>
                    </>
                  ) : (
                    <>
                      <td className="px-4 py-2 text-textPrimary font-medium">{pos.pair}</td>
                      <td className={`px-4 py-2 ${pos.side === 'LONG' ? 'text-bull' : 'text-bear'}`}>
                        {pos.side}
                      </td>
                      <td className="px-4 py-2 text-textPrimary">{formatQty(pos.qty)}</td>
                      <td className="px-4 py-2 text-textPrimary">${formatPrice(pos.entry_price)}</td>
                      <td className="px-4 py-2 text-textPrimary">${formatPrice(pos.close_price)}</td>
                      <td className="px-4 py-2">
                        <span className={`px-1.5 py-0.5 rounded text-xs ${pos.close_type === 'tp_hit' ? 'bg-bull/20 text-bull' :
                          pos.close_type === 'sl_hit' ? 'bg-bear/20 text-bear' :
                            'bg-panelAlt text-textSecondary'
                          }`}>
                          {pos.close_type || 'manual'}
                        </span>
                      </td>
                      <td className={`px-4 py-2 ${pos.pnl_usdt >= 0 ? 'text-bull' : 'text-bear'}`}>
                        {pos.pnl_usdt >= 0 ? '+' : ''}${pos.pnl_usdt?.toFixed(2)}
                      </td>
                      <td className="px-4 py-2 text-textSecondary">
                        {pos.close_time ? new Date(pos.close_time).toLocaleString() : '-'}
                      </td>
                    </>
                  )}
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={10} className="px-4 py-8 text-center text-textSecondary">
                  No {positionsView === 'open' ? 'open' : 'closed'} positions
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
