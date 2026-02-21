import { useState, useEffect } from 'react';
import { useUiStore } from '../../store';
import { X } from 'lucide-react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../../api';
import type { OrderOptions } from '../../types';

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
            signal_score_min: 0.70,
            signal_score_max: 0.80,
            max_hold_bars: 25,
            tf_1h_pct: 70,
            tf_4h_pct: 20,
            tf_15m_pct: 10,
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

    const tfSum = settings.order_manager.tf_1h_pct + settings.order_manager.tf_4h_pct + settings.order_manager.tf_15m_pct;

    if (!orderOptionsModalOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-panel border border-border rounded-xl w-full max-w-lg p-6 max-h-[85vh] overflow-y-auto">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-lg font-semibold text-textPrimary">Order Options</h2>
                    <button className="p-1 hover:bg-panelAlt rounded" onClick={() => setOrderOptionsModalOpen(false)}>
                        <X className="w-5 h-5 text-textSecondary" />
                    </button>
                </div>

                {/* Order Manager Section */}
                <div className="mb-6">
                    <h3 className="text-sm font-medium text-binanceYellow mb-3">Order Manager</h3>
                    <div className="space-y-3">
                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="text-xs text-textSecondary block mb-1">Signal Score Min</label>
                                <input type="number" min="0" max="1" step="0.01"
                                    value={settings.order_manager.signal_score_min}
                                    onChange={(e) => setSettings((s) => ({
                                        ...s,
                                        order_manager: { ...s.order_manager, signal_score_min: parseFloat(e.target.value) || 0 }
                                    }))}
                                    className="input-binance w-full" />
                            </div>
                            <div>
                                <label className="text-xs text-textSecondary block mb-1">Signal Score Max</label>
                                <input type="number" min="0" max="1" step="0.01"
                                    value={settings.order_manager.signal_score_max}
                                    onChange={(e) => setSettings((s) => ({
                                        ...s,
                                        order_manager: { ...s.order_manager, signal_score_max: parseFloat(e.target.value) || 0 }
                                    }))}
                                    className="input-binance w-full" />
                            </div>
                        </div>

                        <div>
                            <label className="text-xs text-textSecondary block mb-1">Max Hold Bars</label>
                            <input type="number" min="1" max="200"
                                value={settings.order_manager.max_hold_bars}
                                onChange={(e) => setSettings((s) => ({
                                    ...s,
                                    order_manager: { ...s.order_manager, max_hold_bars: parseInt(e.target.value) || 1 }
                                }))}
                                className="input-binance w-full" />
                        </div>

                        <div>
                            <label className="text-xs text-textSecondary block mb-1">
                                Timeframe Proportions (sum = 100%)
                                {tfSum !== 100 && <span className="text-bear ml-2">Current: {tfSum}%</span>}
                            </label>
                            <div className="grid grid-cols-3 gap-2">
                                <div>
                                    <label className="text-xs text-textSecondary block mb-1">1h %</label>
                                    <input type="number" min="0" max="100"
                                        value={settings.order_manager.tf_1h_pct}
                                        onChange={(e) => setSettings((s) => ({
                                            ...s,
                                            order_manager: { ...s.order_manager, tf_1h_pct: parseInt(e.target.value) || 0 }
                                        }))}
                                        className="input-binance w-full" />
                                </div>
                                <div>
                                    <label className="text-xs text-textSecondary block mb-1">4h %</label>
                                    <input type="number" min="0" max="100"
                                        value={settings.order_manager.tf_4h_pct}
                                        onChange={(e) => setSettings((s) => ({
                                            ...s,
                                            order_manager: { ...s.order_manager, tf_4h_pct: parseInt(e.target.value) || 0 }
                                        }))}
                                        className="input-binance w-full" />
                                </div>
                                <div>
                                    <label className="text-xs text-textSecondary block mb-1">15m %</label>
                                    <input type="number" min="0" max="100"
                                        value={settings.order_manager.tf_15m_pct}
                                        onChange={(e) => setSettings((s) => ({
                                            ...s,
                                            order_manager: { ...s.order_manager, tf_15m_pct: parseInt(e.target.value) || 0 }
                                        }))}
                                        className="input-binance w-full" />
                                </div>
                            </div>
                        </div>
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
