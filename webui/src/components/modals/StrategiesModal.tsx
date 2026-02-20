import { useUiStore } from '../../store';
import { X } from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../../api';

export default function StrategiesModal() {
  const { strategiesModalOpen, setStrategiesModalOpen } = useUiStore();
  const queryClient = useQueryClient();

  const { data: strategies = [] } = useQuery({
    queryKey: ['strategies'],
    queryFn: () => apiService.getStrategies(),
    enabled: strategiesModalOpen,
  });

  const toggleMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) => apiService.toggleStrategy(id, enabled),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['strategies'] }); },
  });

  if (!strategiesModalOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-panel border border-border rounded-xl w-full max-w-2xl p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-textPrimary">Strategies</h2>
          <button className="p-1 hover:bg-panelAlt rounded" onClick={() => setStrategiesModalOpen(false)}>
            <X className="w-5 h-5 text-textSecondary" />
          </button>
        </div>

        <div className="space-y-3">
          {strategies.map((strategy: any) => (
            <div key={strategy.id} className="p-4 bg-panelAlt border border-border rounded-lg">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="text-sm font-medium text-textPrimary">{strategy.name}</h3>
                  <p className="text-xs text-textSecondary mt-1">{strategy.description}</p>
                </div>
                <label className="flex items-center gap-2 cursor-pointer">
                  <span className="text-xs text-textSecondary">{strategy.enabled ? 'On' : 'Off'}</span>
                  <div className={`w-10 h-5 rounded-full relative transition-colors ${strategy.enabled ? 'bg-binanceYellow' : 'bg-border'}`}>
                    <input type="checkbox" className="sr-only" checked={strategy.enabled}
                      onChange={(e) => toggleMutation.mutate({ id: strategy.id, enabled: e.target.checked })} />
                    <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${strategy.enabled ? 'translate-x-5' : 'translate-x-0.5'}`} />
                  </div>
                </label>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-end">
          <button className="btn-binance btn-binance-secondary" onClick={() => setStrategiesModalOpen(false)}>Close</button>
        </div>
      </div>
    </div>
  );
}
