import { useState, useEffect } from 'react';
import { useUiStore } from '../../store';
import { X } from 'lucide-react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../../api';
import type { OrderOptions } from '../../types';

interface TimeframeRow {
    tf: string;
    tfMinutes: number;
    scoreMin: number;
    scoreMax: number;
    pct: number;
    pctKey: keyof OrderOptions['order_manager'];
    minKey: keyof OrderOptions['order_manager'];
    maxKey: keyof OrderOptions['order_manager'];
}

const TIMEFRAMES: TimeframeRow[] = [
    { tf: '1m', tfMinutes: 1, scoreMin: 0.70, scoreMax: 0.80, pct: 0, pctKey: 'tf_1m_pct', minKey: 'signal_score_min_1m', maxKey: 'signal_score_max_1m' },
    { tf: '5m', tfMinutes: 5, scoreMin: 0.70, scoreMax: 0.80, pct: 0, pctKey: 'tf_5m_pct', minKey: 'signal_score_min_5m', maxKey: 'signal_score_max_5m' },
    { tf: '15m', tfMinutes: 15, scoreMin: 0.70, scoreMax: 0.80, pct: 0, pctKey: 'tf_15m_pct', minKey: 'signal_score_min_15m', maxKey: 'signal_score_max_15m' },
    { tf: '1h', tfMinutes: 60, scoreMin: 0.70, scoreMax: 0.80, pct: 0, pctKey: 'tf_1h_pct', minKey: 'signal_score_min_1h', maxKey: 'signal_score_max_1h' },
    { tf: '4h', tfMinutes: 240, scoreMin: 0.70, scoreMax: 0.80, pct: 0, pctKey: 'tf_4h_pct', minKey: 'signal_score_min_4h', maxKey: 'signal_score_max_4h' },
    { tf: '1d', tfMinutes: 1440, scoreMin: 0.70, scoreMax: 0.80, pct: 0, pctKey: 'tf_1d_pct', minKey: 'signal_score_min_1d', maxKey: 'signal_score_max_1d' },
];

export default function OrderOptionsModal() {
    const { orderOptionsModalOpen, setOrderOptionsModalOpen } = useUiStore();
    const queryClient = useQueryClient();

    const { data: loadedOptions } = useQuery({
        queryKey: ['order-options'],
        queryFn: () => apiService.getOrderOptions(),
        enabled: orderOptionsModalOpen,
    });

    const [settings, setSettings] = useState<OrderOptions>({
        order_manager: {
            signal_score_min_1m: 0.70,
            signal_score_max_1m: 0.80,
            signal_score_min_5m: 0.70,
            signal_score_max_5m: 0.80,
            signal_score_min_15m: 0.70,
            signal_score_max_15m: 0.80,
            signal_score_min_1h: 0.70,
            signal_score_max_1h: 0.80,
            signal_score_min_4h: 0.70,
            signal_score_max_4h: 0.80,
            signal_score_min_1d: 0.70,
            signal_score_max_1d: 0.80,
            max_hold_bars: 25,
            tf_1m_pct: 0,
            tf_5m_pct: 0,
            tf_15m_pct: 10,
            tf_1h_pct: 70,
            tf_4h_pct: 20,
            tf_1d_pct: 0,
        },
        risk_manager: {
            btc_alert_threshold_pct: 0.5,
            alt_alert_threshold_pct: 1.5,
            volume_spike_threshold: 2.0,
        },
    });

    useEffect(() => {
        if (loadedOptions) setSettings(loadedOptions);
    }, [loadedOptions]);

    const saveMutation = useMutation({
        mutationFn: apiService.saveOrderOptions,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['order-options'] });
            setOrderOptionsModalOpen(false);
        },
    });

    const tfSum = settings.order_manager.tf_1m_pct + 
                  settings.order_manager.tf_5m_pct + 
                  settings.order_manager.tf_15m_pct + 
                  settings.order_manager.tf_1h_pct + 
                  settings.order_manager.tf_4h_pct + 
                  settings.order_manager.tf_1d_pct;

    if (!orderOptionsModalOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-panel border border-border rounded-xl w-full max-w-2xl p-6 max-h-[90vh] overflow-y-auto">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-lg font-semibold text-textPrimary">Order Options</h2>
                    <button className="p-1 hover:bg-panelAlt rounded" onClick={() => setOrderOptionsModalOpen(false)}>
                        <X className="w-5 h-5 text-textSecondary" />
                    </button>
                </div>

                {/* Order Manager Section */}
                <div className="mb-6">
                    <h3 className="text-sm font-medium text-binanceYellow mb-3">Order Manager - Timeframe & Score Settings</h3>
                    
                    {/* Timeframe Table */}
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-border">
                                    <th className="text-left py-2 px-2 text-textSecondary font-medium">Timeframe</th>
                                    <th className="text-left py-2 px-2 text-textSecondary font-medium">Score Min</th>
                                    <th className="text-left py-2 px-2 text-textSecondary font-medium">Score Max</th>
                                    <th className="text-left py-2 px-2 text-textSecondary font-medium">Order %</th>
                                </tr>
                            </thead>
                            <tbody>
                                {TIMEFRAMES.map((row) => {
                                    const scoreMin = settings.order_manager[row.minKey as keyof typeof settings.order_manager] as number;
                                    const scoreMax = settings.order_manager[row.maxKey as keyof typeof settings.order_manager] as number;
                                    const pct = settings.order_manager[row.pctKey as keyof typeof settings.order_manager] as number;
                                    
                                    return (
                                        <tr key={row.tf} className="border-b border-border/50">
                                            <td className="py-2 px-2 text-textPrimary font-medium">{row.tf}</td>
                                            <td className="py-2 px-2">
                                                <input type="number" min="0" max="1" step="0.01"
                                                    value={scoreMin}
                                                    onChange={(e) => setSettings((s) => ({
                                                        ...s,
                                                        order_manager: { ...s.order_manager, [row.minKey]: parseFloat(e.target.value) || 0 }
                                                    }))}
                                                    className="input-binance w-full text-xs" />
                                            </td>
                                            <td className="py-2 px-2">
                                                <input type="number" min="0" max="1" step="0.01"
                                                    value={scoreMax}
                                                    onChange={(e) => setSettings((s) => ({
                                                        ...s,
                                                        order_manager: { ...s.order_manager, [row.maxKey]: parseFloat(e.target.value) || 0 }
                                                    }))}
                                                    className="input-binance w-full text-xs" />
                                            </td>
                                            <td className="py-2 px-2">
                                                <input type="number" min="0" max="100"
                                                    value={pct}
                                                    onChange={(e) => setSettings((s) => ({
                                                        ...s,
                                                        order_manager: { ...s.order_manager, [row.pctKey]: parseInt(e.target.value) || 0 }
                                                    }))}
                                                    className="input-binance w-full text-xs" />
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                    
                    <div className={`text-xs mt-2 ${tfSum !== 100 ? 'text-bear' : 'text-textSecondary'}`}>
                        Total: {tfSum}% (must equal 100%)
                    </div>

                    <div className="mt-4">
                        <label className="text-xs text-textSecondary block mb-1">Max Hold Bars</label>
                        <input type="number" min="1" max="200"
                            value={settings.order_manager.max_hold_bars}
                            onChange={(e) => setSettings((s) => ({
                                ...s,
                                order_manager: { ...s.order_manager, max_hold_bars: parseInt(e.target.value) || 1 }
                            }))}
                            className="input-binance w-full" />
                    </div>
                </div>

                {/* Risk Manager Section */}
                <div className="mb-6 pt-4 border-t border-border">
                    <h3 className="text-sm font-medium text-binanceYellow mb-3">Risk Manager</h3>
                    <div className="space-y-3">
                        <div>
                            <label className="text-xs text-textSecondary block mb-1">BTC Alert Threshold (%)</label>
                            <input type="number" min="0.1" max="10" step="0.1"
                                value={settings.risk_manager.btc_alert_threshold_pct}
                                onChange={(e) => setSettings((s) => ({
                                    ...s,
                                    risk_manager: { ...s.risk_manager, btc_alert_threshold_pct: parseFloat(e.target.value) || 0.5 }
                                }))}
                                className="input-binance w-full" />
                        </div>
                        <div>
                            <label className="text-xs text-textSecondary block mb-1">Alt Alert Threshold (%)</label>
                            <input type="number" min="0.1" max="20" step="0.1"
                                value={settings.risk_manager.alt_alert_threshold_pct}
                                onChange={(e) => setSettings((s) => ({
                                    ...s,
                                    risk_manager: { ...s.risk_manager, alt_alert_threshold_pct: parseFloat(e.target.value) || 1.5 }
                                }))}
                                className="input-binance w-full" />
                        </div>
                        <div>
                            <label className="text-xs text-textSecondary block mb-1">Volume Spike Threshold (x)</label>
                            <input type="number" min="0.5" max="10" step="0.1"
                                value={settings.risk_manager.volume_spike_threshold}
                                onChange={(e) => setSettings((s) => ({
                                    ...s,
                                    risk_manager: { ...s.risk_manager, volume_spike_threshold: parseFloat(e.target.value) || 2.0 }
                                }))}
                                className="input-binance w-full" />
                        </div>
                    </div>
                </div>

                <div className="flex gap-2">
                    <button className="flex-1 btn-binance btn-binance-secondary" onClick={() => setOrderOptionsModalOpen(false)}>Cancel</button>
                    <button
                        className={`flex-1 btn-binance btn-binance-primary ${tfSum !== 100 ? 'opacity-50 cursor-not-allowed' : ''}`}
                        onClick={() => {
                            if (tfSum === 100) saveMutation.mutate(settings);
                        }}
                        disabled={tfSum !== 100}
                    >
                        {saveMutation.isPending ? 'Saving...' : 'Save'}
                    </button>
                </div>
            </div>
        </div>
    );
}
