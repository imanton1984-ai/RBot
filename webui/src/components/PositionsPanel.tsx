import { useUiStore, useDataStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { Settings } from 'lucide-react';

export default function PositionsPanel() {
  const { positionsView, setPositionsView } = useUiStore();
  const { positions, setPositions } = useDataStore();

  // Load positions only when tab is active - refetch every 5 seconds
  const { data: openPositions } = useQuery({
    queryKey: ['positions-open'],
    queryFn: () => apiService.getOpenPositions(),
    refetchInterval: 5000,
    staleTime: 3000,
    enabled: positionsView === 'open',
    retry: 1,
  });

  const { data: historyPositions } = useQuery({
    queryKey: ['positions-history'],
    queryFn: () => apiService.getPositionsHistory(),
    refetchInterval: 10000,
    staleTime: 5000,
    enabled: positionsView === 'history',
    retry: 1,
  });

  // Update positions when data arrives
  if (positionsView === 'open' && openPositions) {
    setPositions(openPositions);
  }

  const currentPositions = positionsView === 'open' ? positions : (historyPositions || []);

  return (
    <div className="h-[40%] min-h-[200px] border-t border-border flex flex-col shrink-0">
      <div className="h-10 border-b border-border flex items-center px-4 gap-2">
        <button
          className={`px-3 py-1.5 rounded text-sm font-medium ${
            positionsView === 'open'
              ? 'bg-binanceYellow text-black'
              : 'text-textSecondary hover:bg-panelAlt'
          }`}
          onClick={() => setPositionsView('open')}
        >
          Open Positions
        </button>
        <button
          className={`px-3 py-1.5 rounded text-sm font-medium ${
            positionsView === 'history'
              ? 'bg-binanceYellow text-black'
              : 'text-textSecondary hover:bg-panelAlt'
          }`}
          onClick={() => setPositionsView('history')}
        >
          History
        </button>

        <div className="flex-1" />

        <button className="p-1.5 hover:bg-panelAlt rounded">
          <Settings className="w-4 h-4 text-textSecondary" />
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
                <th className="text-left font-normal px-4 py-2">Candles Left</th>
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
                >
                  {positionsView === 'open' ? (
                    <>
                      <td className="px-4 py-2 text-textPrimary font-medium">{pos.pair}</td>
                      <td className={`px-4 py-2 ${pos.side === 'LONG' ? 'text-bull' : 'text-bear'}`}>
                        {pos.side}
                      </td>
                      <td className="px-4 py-2 text-textPrimary">{pos.qty?.toFixed(4)}</td>
                      <td className="px-4 py-2 text-textPrimary">${pos.entry_price?.toFixed(2)}</td>
                      <td className="px-4 py-2 text-textPrimary">${pos.current_price?.toFixed(2)}</td>
                      <td className="px-4 py-2 text-textPrimary">${pos.stop_loss?.toFixed(2) || '-'}</td>
                      <td className="px-4 py-2 text-textPrimary">${pos.take_profit?.toFixed(2) || '-'}</td>
                      <td className={`px-4 py-2 ${pos.pnl_usdt >= 0 ? 'text-bull' : 'text-bear'}`}>
                        ${pos.pnl_usdt?.toFixed(2)} ({pos.pnl_pct?.toFixed(2)}%)
                      </td>
                      <td className="px-4 py-2 text-textPrimary">
                        {pos.candles_left ?? 10}
                      </td>
                    </>
                  ) : (
                    <>
                      <td className="px-4 py-2 text-textPrimary font-medium">{pos.pair}</td>
                      <td className={`px-4 py-2 ${pos.side === 'LONG' ? 'text-bull' : 'text-bear'}`}>
                        {pos.side}
                      </td>
                      <td className="px-4 py-2 text-textPrimary">{pos.qty?.toFixed(4)}</td>
                      <td className="px-4 py-2 text-textPrimary">${pos.entry_price?.toFixed(2)}</td>
                      <td className="px-4 py-2 text-textPrimary">${pos.close_price?.toFixed(2)}</td>
                      <td className="px-4 py-2 text-textSecondary">{pos.close_type || 'manual'}</td>
                      <td className={`px-4 py-2 ${pos.pnl_usdt >= 0 ? 'text-bull' : 'text-bear'}`}>
                        ${pos.pnl_usdt?.toFixed(2)}
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
                <td colSpan={9} className="px-4 py-8 text-center text-textSecondary">
                  No {positionsView} positions
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
