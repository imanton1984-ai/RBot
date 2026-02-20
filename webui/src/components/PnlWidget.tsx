import { useEffect } from 'react';
import { useDataStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';

export default function PnlWidget() {
  const pnlOverview = useDataStore((s) => s.pnlOverview);
  const setPnlOverview = useDataStore((s) => s.setPnlOverview);

  // Load PnL overview every 10 seconds
  const { data } = useQuery({
    queryKey: ['pnl-overview'],
    queryFn: () => apiService.getPnlOverview('1d'),
    refetchInterval: 10000,
    staleTime: 5000,
    retry: 1,
  });

  // Sync to store inside useEffect to avoid infinite re-render loop
  useEffect(() => {
    if (data) setPnlOverview(data);
  }, [data, setPnlOverview]);

  const pnl = data || pnlOverview;

  return (
    <div className="card m-3 mb-0">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-textPrimary">PnL Overview</h3>
      </div>

      <div className="space-y-2 text-xs">
        <div className="flex justify-between items-center">
          <span className="text-textSecondary">Closed PnL:</span>
          <span className={pnl?.closed_pnl >= 0 ? 'text-bull' : 'text-bear'}>
            +${pnl?.closed_pnl?.toFixed(2) ?? '0.00'}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-textSecondary">Unrealized:</span>
          <span className={pnl?.unrealized_pnl >= 0 ? 'text-bull' : 'text-bear'}>
            ${pnl?.unrealized_pnl?.toFixed(2) ?? '0.00'}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-textSecondary">Win Rate:</span>
          <span className={pnl?.win_rate >= 50 ? 'text-bull' : 'text-bear'}>
            {pnl?.win_rate?.toFixed(0) ?? '0'}%
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-textSecondary">Today Trades:</span>
          <span className="text-textPrimary">{pnl?.today_trades ?? 0}</span>
        </div>
      </div>
    </div>
  );
}
