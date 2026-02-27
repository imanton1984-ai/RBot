import { useEffect, useMemo } from 'react';
import { useDataStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { TrendingUp } from 'lucide-react';

export default function PnlWidget() {
  const pnlOverview = useDataStore((s) => s.pnlOverview);
  const setPnlOverview = useDataStore((s) => s.setPnlOverview);
  const positions = useDataStore((s) => s.positions);

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

  // Derive unrealized PnL from open positions (updates every 2s via positions polling)
  // This is faster than the /api/pnl/overview endpoint (10s)
  const liveUnrealizedPnl = useMemo(() => {
    if (!positions || positions.length === 0) return pnl?.unrealized_pnl ?? 0;
    return positions.reduce((sum: number, pos: any) => sum + (pos.pnl_usdt ?? 0), 0);
  }, [positions, pnl?.unrealized_pnl]);

  // Total PnL = closed + live unrealized (all without leverage — real USDT values)
  const totalPnl = (pnl?.closed_pnl ?? 0) + liveUnrealizedPnl;
  // PnL % calculated from overall balance (wallet_balance, no leverage)
  const overallBalance = pnl?.overall_balance ?? 0;
  const pnlPct = overallBalance > 0 ? (totalPnl / overallBalance) * 100 : 0;
  const closedPnlPct = overallBalance > 0 ? ((pnl?.closed_pnl ?? 0) / overallBalance) * 100 : 0;
  const unrealizedPnlPct = overallBalance > 0 ? (liveUnrealizedPnl / overallBalance) * 100 : 0;

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
          <div className="text-right">
            <span className={`text-sm font-bold ${totalPnl >= 0 ? 'text-bull' : 'text-bear'}`}>
              {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
            </span>
          </div>
        </div>

        <div className="pt-2 border-t border-border space-y-1.5">
          <div className="flex justify-between items-center">
            <span className="text-textSecondary">Closed PnL:</span>
            <div className="flex items-center gap-1.5">
              <span className={(pnl?.closed_pnl ?? 0) >= 0 ? 'text-bull' : 'text-bear'}>
                {(pnl?.closed_pnl ?? 0) >= 0 ? '+' : ''}${pnl?.closed_pnl?.toFixed(2) ?? '0.00'}
              </span>
              <span className={`text-[10px] ${closedPnlPct >= 0 ? 'text-bull/70' : 'text-bear/70'}`}>
                ({closedPnlPct >= 0 ? '+' : ''}{closedPnlPct.toFixed(2)}%)
              </span>
            </div>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-textSecondary">Unrealized:</span>
            <div className="flex items-center gap-1.5">
              <span className={liveUnrealizedPnl >= 0 ? 'text-bull' : 'text-bear'}>
                {liveUnrealizedPnl >= 0 ? '+' : ''}${liveUnrealizedPnl.toFixed(2)}
              </span>
              <span className={`text-[10px] ${unrealizedPnlPct >= 0 ? 'text-bull/70' : 'text-bear/70'}`}>
                ({unrealizedPnlPct >= 0 ? '+' : ''}{unrealizedPnlPct.toFixed(2)}%)
              </span>
            </div>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-textSecondary">Win Rate:</span>
            <span className={(pnl?.win_rate ?? 0) >= 50 ? 'text-bull' : 'text-bear'}>
              {pnl?.win_rate?.toFixed(1) ?? '0.0'}%
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
