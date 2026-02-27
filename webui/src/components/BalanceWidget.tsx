import { useEffect, useMemo } from 'react';
import { useDataStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { Wallet } from 'lucide-react';

export default function BalanceWidget() {
  const { balance, setBalance } = useDataStore();
  const positions = useDataStore((s) => s.positions);

  const { data } = useQuery({
    queryKey: ['balance'],
    queryFn: () => apiService.getBalance(),
    refetchInterval: 15000,
    staleTime: 10000,
  });

  useEffect(() => {
    if (data) setBalance(data);
  }, [data, setBalance]);

  const bal = balance || data || { overall: 0, in_orders: 0, available: 0, wallet_balance: 0, unrealized_pnl: 0 };

  // Derive live unrealized PnL from open positions (updates every 2s) — synced with PositionsPanel
  const liveUnrealizedPnl = useMemo(() => {
    if (!positions || positions.length === 0) return bal.unrealized_pnl ?? 0;
    return positions.reduce((sum: number, pos: any) => sum + (pos.pnl_usdt ?? 0), 0);
  }, [positions, bal.unrealized_pnl]);

  return (
    <div className="card m-3 mb-0">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Wallet className="w-4 h-4 text-textSecondary" />
          <h3 className="text-sm font-medium text-textPrimary">Balance</h3>
        </div>
      </div>

      <div className="space-y-3">
        {/* Overall = wallet_balance (real deposit, no leverage) */}
        <div className="flex justify-between items-center">
          <span className="text-textSecondary text-xs">Overall</span>
          <span className="text-lg font-bold text-binanceYellow">
            ${bal.overall?.toFixed(2) ?? '0.00'}
          </span>
        </div>

        <div className="pt-3 border-t border-border space-y-2">
          {/* Wallet Balance */}
          <div className="flex justify-between items-center text-xs">
            <span className="text-textSecondary">Wallet Balance</span>
            <span className="text-textPrimary">${bal.wallet_balance?.toFixed(2) ?? '0.00'}</span>
          </div>
          {/* In Orders = real margin used (notional / leverage) */}
          <div className="flex justify-between items-center text-xs">
            <span className="text-textSecondary">In Orders (Margin)</span>
            <span className="text-textPrimary">${bal.in_orders?.toFixed(2) ?? '0.00'}</span>
          </div>
          {/* Available */}
          <div className="flex justify-between items-center text-xs">
            <span className="text-textSecondary">Available</span>
            <span className="text-textPrimary">${bal.available?.toFixed(2) ?? '0.00'}</span>
          </div>
          {/* Unrealized PnL — synced with Open Positions (fastest source) */}
          <div className="flex justify-between items-center text-xs">
            <span className="text-textSecondary">Unrealized PnL</span>
            <span className={liveUnrealizedPnl >= 0 ? 'text-bull' : 'text-bear'}>
              {liveUnrealizedPnl >= 0 ? '+' : ''}${liveUnrealizedPnl.toFixed(2)}
              {bal.wallet_balance > 0 && (
                <span className="text-[10px] ml-1 opacity-70">
                  ({liveUnrealizedPnl >= 0 ? '+' : ''}{((liveUnrealizedPnl / bal.wallet_balance) * 100).toFixed(2)}%)
                </span>
              )}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
