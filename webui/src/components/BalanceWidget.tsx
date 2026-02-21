import { useEffect } from 'react';
import { useDataStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { Wallet } from 'lucide-react';

export default function BalanceWidget() {
  const { balance, setBalance } = useDataStore();

  const { data } = useQuery({
    queryKey: ['balance'],
    queryFn: () => apiService.getBalance(),
    refetchInterval: 30000,
  });

  useEffect(() => {
    if (data) setBalance(data);
  }, [data]);

  const bal = balance || data || { overall: 0, in_orders: 0, available: 0, wallet_balance: 0, unrealized_pnl: 0 };

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
          {/* Unrealized PnL */}
          <div className="flex justify-between items-center text-xs">
            <span className="text-textSecondary">Unrealized PnL</span>
            <span className={bal.unrealized_pnl >= 0 ? 'text-bull' : 'text-bear'}>
              {bal.unrealized_pnl >= 0 ? '+' : ''}${bal.unrealized_pnl?.toFixed(2) ?? '0.00'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
