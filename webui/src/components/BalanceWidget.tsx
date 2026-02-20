import { useEffect } from 'react';
import { useDataStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { ChevronDown } from 'lucide-react';

export default function BalanceWidget() {
  const { balance, setBalance } = useDataStore();

  const { data } = useQuery({
    queryKey: ['balance'],
    queryFn: () => apiService.getBalance(),
    refetchInterval: 5000,
  });

  useEffect(() => {
    if (data) setBalance(data);
  }, [data]);

  const bal = balance || data || { overall: 0, in_orders: 0, available: 0 };

  return (
    <div className="card m-3 mb-0">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-textPrimary">Balance</h3>
        <button className="text-xs text-binanceYellow flex items-center gap-1">
          View All <ChevronDown className="w-3 h-3" />
        </button>
      </div>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-textSecondary text-xs">Overall</span>
          <span className="text-lg font-bold text-binanceYellow">${bal.overall?.toFixed(2) ?? '0.00'}</span>
        </div>

        <div className="pt-3 border-t border-border space-y-2">
          <div className="flex justify-between items-center text-xs">
            <span className="text-textSecondary">Balance</span>
            <span className="text-textPrimary">${bal.overall?.toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center text-xs">
            <span className="text-textSecondary">In Orders</span>
            <span className="text-textPrimary">${bal.in_orders?.toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center text-xs">
            <span className="text-textSecondary">Available</span>
            <span className="text-textPrimary">${bal.available?.toFixed(2)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
