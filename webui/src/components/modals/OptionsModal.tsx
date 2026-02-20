import { useState } from 'react';
import { useUiStore } from '../../store';
import { X } from 'lucide-react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../../api';

export default function OptionsModal() {
  const { optionsModalOpen, setOptionsModalOpen } = useUiStore();
  const [settings, setSettings] = useState({
    leverage_default: 10,
    max_open_orders: 10,
    max_risk_pct: 2.0,
    order_timeout_bars: 10,
    ws_update_rate_ms: 100,
  });

  const queryClient = useQueryClient();
  
  const saveMutation = useMutation({
    mutationFn: apiService.saveOptions,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['options'] });
      setOptionsModalOpen(false);
    },
  });

  if (!optionsModalOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-panel border border-border rounded-xl w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-textPrimary">Trading Options</h2>
          <button className="p-1 hover:bg-panelAlt rounded" onClick={() => setOptionsModalOpen(false)}>
            <X className="w-5 h-5 text-textSecondary" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-xs text-textSecondary block mb-1">Default Leverage</label>
            <input type="number" min="1" max="125" value={settings.leverage_default}
              onChange={(e) => setSettings((s) => ({ ...s, leverage_default: parseInt(e.target.value) || 1 }))}
              className="input-binance w-full" />
          </div>
          <div>
            <label className="text-xs text-textSecondary block mb-1">Max Open Orders</label>
            <input type="number" min="1" max="100" value={settings.max_open_orders}
              onChange={(e) => setSettings((s) => ({ ...s, max_open_orders: parseInt(e.target.value) || 1 }))}
              className="input-binance w-full" />
          </div>
          <div>
            <label className="text-xs text-textSecondary block mb-1">Max Risk (%)</label>
            <input type="number" min="0.1" max="100" step="0.1" value={settings.max_risk_pct}
              onChange={(e) => setSettings((s) => ({ ...s, max_risk_pct: parseFloat(e.target.value) || 1 }))}
              className="input-binance w-full" />
          </div>
          <div>
            <label className="text-xs text-textSecondary block mb-1">Order Timeout (bars)</label>
            <input type="number" min="1" value={settings.order_timeout_bars}
              onChange={(e) => setSettings((s) => ({ ...s, order_timeout_bars: parseInt(e.target.value) || 1 }))}
              className="input-binance w-full" />
          </div>
        </div>

        <div className="flex gap-2 mt-6">
          <button className="flex-1 btn-binance btn-binance-secondary" onClick={() => setOptionsModalOpen(false)}>Cancel</button>
          <button className="flex-1 btn-binance btn-binance-primary" onClick={() => saveMutation.mutate(settings)}>Save</button>
        </div>
      </div>
    </div>
  );
}
