import { useState, useMemo } from 'react';
import { useUiStore, useTradingStore } from '../../store';
import { X, Search } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../../api';

export default function SignalsModal() {
  const { signalsModalOpen, setSignalsModalOpen } = useUiStore();
  const { setCurrentPair, setCurrentTf } = useTradingStore();
  const [filters, setFilters] = useState({ pair: '', tf: '', status: 'all' });

  const { data: signals = [] } = useQuery({
    queryKey: ['signals', filters.pair, filters.tf],
    queryFn: () => apiService.getSignals(filters.pair || undefined, filters.tf ? parseInt(filters.tf) : undefined, 100),
    enabled: signalsModalOpen,
  });

  const filteredSignals = useMemo(() => {
    return signals.filter((s: any) => {
      if (filters.status !== 'all' && s.status !== filters.status) return false;
      return true;
    });
  }, [signals, filters.status]);

  const handleSignalClick = (signal: any) => {
    setCurrentPair(signal.pair);
    setCurrentTf(signal.tf);
    setSignalsModalOpen(false);
  };

  if (!signalsModalOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-panel border border-border rounded-xl w-full max-w-4xl p-6 max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between mb-4 shrink-0">
          <h2 className="text-lg font-semibold text-textPrimary">Trade Signals</h2>
          <button className="p-1 hover:bg-panelAlt rounded" onClick={() => setSignalsModalOpen(false)}>
            <X className="w-5 h-5 text-textSecondary" />
          </button>
        </div>

        <div className="flex items-center gap-2 mb-4 shrink-0">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-textSecondary" />
            <input type="text" placeholder="Filter by pair..." value={filters.pair}
              onChange={(e) => setFilters((f) => ({ ...f, pair: e.target.value }))}
              className="input-binance w-full pl-9" />
          </div>
          <select value={filters.tf} onChange={(e) => setFilters((f) => ({ ...f, tf: e.target.value }))} className="input-binance">
            <option value="">All TF</option>
            <option value="1">1m</option><option value="5">5m</option><option value="15">15m</option>
            <option value="60">1h</option><option value="240">4h</option><option value="1440">1d</option>
          </select>
          <select value={filters.status} onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value }))} className="input-binance">
            <option value="all">All</option><option value="active">Active</option><option value="closed">Closed</option>
          </select>
        </div>

        <div className="flex-1 overflow-auto min-h-0">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-panel text-textSecondary">
              <tr>
                <th className="text-left font-normal px-3 py-2">Time</th>
                <th className="text-left font-normal px-3 py-2">Pair</th>
                <th className="text-left font-normal px-3 py-2">TF</th>
                <th className="text-left font-normal px-3 py-2">Side</th>
                <th className="text-left font-normal px-3 py-2">Score</th>
                <th className="text-left font-normal px-3 py-2">Entry</th>
                <th className="text-left font-normal px-3 py-2">SL</th>
                <th className="text-left font-normal px-3 py-2">TP</th>
                <th className="text-left font-normal px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredSignals.map((signal: any) => (
                <tr key={signal.id} className="border-t border-border hover:bg-panelAlt cursor-pointer transition-colors" onClick={() => handleSignalClick(signal)}>
                  <td className="px-3 py-2 text-textSecondary">{new Date(signal.time).toLocaleString()}</td>
                  <td className="px-3 py-2 text-textPrimary font-medium">{signal.pair}</td>
                  <td className="px-3 py-2 text-textSecondary">{signal.tf >= 1440 ? `${signal.tf/1440}d` : signal.tf >= 60 ? `${signal.tf/60}h` : `${signal.tf}m`}</td>
                  <td className={`px-3 py-2 ${signal.side === 1 ? 'text-bull' : 'text-bear'}`}>{signal.side === 1 ? 'LONG' : 'SHORT'}</td>
                  <td className="px-3 py-2"><span className={`px-2 py-0.5 rounded text-xs ${signal.score >= 0.8 ? 'bg-bull text-white' : 'bg-border text-textSecondary'}`}>{signal.score.toFixed(2)}</span></td>
                  <td className="px-3 py-2 text-textPrimary">${signal.entry_price?.toFixed(2)}</td>
                  <td className="px-3 py-2 text-textSecondary">${signal.sl_price?.toFixed(2)}</td>
                  <td className="px-3 py-2 text-textPrimary">${signal.tp_price?.toFixed(2)}</td>
                  <td className="px-3 py-2"><span className={`px-2 py-0.5 rounded text-xs ${signal.status === 'active' ? 'bg-bull text-white' : 'bg-panelAlt text-textSecondary'}`}>{signal.status}</span></td>
                </tr>
              ))}
              {filteredSignals.length === 0 && (
                <tr><td colSpan={8} className="px-3 py-8 text-center text-textSecondary">No signals found</td></tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="mt-4 flex justify-end shrink-0">
          <button className="btn-binance btn-binance-secondary" onClick={() => setSignalsModalOpen(false)}>Close</button>
        </div>
      </div>
    </div>
  );
}
