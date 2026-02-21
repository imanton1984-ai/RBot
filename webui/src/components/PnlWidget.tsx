import { useEffect } from 'react';
import { useDataStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { TrendingUp } from 'lucide-react';

export default function PnlWidget() {
  const pnlOverview = useDataStore((s) => s.pnlOverview);
  const setPnlOverview = useDataStore((s) => s.setPnlOverview);

  const { data } = useQuery({
    queryKey: ['pnl-overview'],
    queryFn: () => apiService.getPnlOverview('30d'),
    refetchInterval: 10000,
    staleTime: 5000,
    retry: 1,
  });

  useEffect(() => {
    if (data) setPnlOverview(data);
  }, [data, setPnlOverview]);

  const pnl = data || pnlOverview;

  // Total PnL = closed + unrealized (all without leverage — real USDT values)
  const totalPnl = (pnl?.closed_pnl ?? 0) + (pnl?.unrealized_pnl ?? 0);
  // PnL % calculated from overall balance (wallet_balance, no leverage)
  const overallBalance = pnl?.overall_balance ?? 0;
  const pnlPct = overallBalance > 0 ? (totalPnl / overallBalance) * 100 : 0;

  return (
    <div className="card m-3 mb-0">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-textSecondary" />
          <h3 className="text-sm font-medium text-textPrimary">PnL Overview</h3>
        </div>
        <span className={`text-xs px-2 py-0.5 rounded ${totalPnl >= 0 ? 'bg-bull/20 text-bull' : 'bg-bear/20 text-bear'}`}>
          {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(2)}%
        </span>
      </div>

      <div className="space-y-2 text-xs">
        {/* Total PnL (without leverage) */}
        <div className="flex justify-between items-center">
          <span className="text-textSecondary">Total PnL:</span>
          <span className={`text-sm font-bold ${totalPnl >= 0 ? 'text-bull' : 'text-bear'}`}>
            {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
          </span>
        </div>

        <div className="pt-2 border-t border-border space-y-1.5">
          <div className="flex justify-between items-center">
            <span className="text-textSecondary">Closed PnL:</span>
            <span className={(pnl?.closed_pnl ?? 0) >= 0 ? 'text-bull' : 'text-bear'}>
              {(pnl?.closed_pnl ?? 0) >= 0 ? '+' : ''}${pnl?.closed_pnl?.toFixed(2) ?? '0.00'}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-textSecondary">Unrealized:</span>
            <span className={(pnl?.unrealized_pnl ?? 0) >= 0 ? 'text-bull' : 'text-bear'}>
              {(pnl?.unrealized_pnl ?? 0) >= 0 ? '+' : ''}${pnl?.unrealized_pnl?.toFixed(2) ?? '0.00'}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-textSecondary">Win Rate:</span>
            <span className={(pnl?.win_rate ?? 0) >= 50 ? 'text-bull' : 'text-bear'}>
              {pnl?.win_rate?.toFixed(0) ?? '0'}%
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-textSecondary">Total Trades:</span>
            <span className="text-textPrimary">{pnl?.today_trades ?? 0}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
