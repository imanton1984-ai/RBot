import { useState, useEffect } from 'react';
import { useUiStore } from '../../store';
import { X } from 'lucide-react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../../api';
import type { TradingOptions } from '../../types';

export default function TradingOptionsModal() {
    const { tradingOptionsModalOpen, setTradingOptionsModalOpen } = useUiStore();
    const queryClient = useQueryClient();

    const { data: loadedOptions } = useQuery({
        queryKey: ['trading-options'],
        queryFn: () => apiService.getTradingOptions(),
        enabled: tradingOptionsModalOpen,
    });

    const [settings, setSettings] = useState<TradingOptions>({
        leverage: 10,
        max_orders_at_a_time: 10,
        trade_size_type: 'fixed_usdt',
        trade_size_value: 100.0,
        strategy_type: 'ml_super_entry',
        order_type: 'futures_oco',
        trading_mode: 'off',
    });

    useEffect(() => {
        if (loadedOptions) {
            setSettings(loadedOptions);
        }
    }, [loadedOptions]);

    const saveMutation = useMutation({
        mutationFn: apiService.saveTradingOptions,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['trading-options'] });
            setTradingOptionsModalOpen(false);
        },
    });

    if (!tradingOptionsModalOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-panel border border-border rounded-xl w-full max-w-md p-6 max-h-[85vh] overflow-y-auto">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-lg font-semibold text-textPrimary">Trading Options</h2>
                    <button className="p-1 hover:bg-panelAlt rounded" onClick={() => setTradingOptionsModalOpen(false)}>
                        <X className="w-5 h-5 text-textSecondary" />
                    </button>
                </div>

                <div className="space-y-4">
                    {/* Leverage */}
                    <div>
                        <label className="text-xs text-textSecondary block mb-1">Leverage: {settings.leverage}x</label>
                        <input type="range" min="1" max="125" value={settings.leverage}
                            onChange={(e) => setSettings((s) => ({ ...s, leverage: parseInt(e.target.value) || 1 }))}
                            className="w-full accent-binanceYellow" />
                        <div className="flex gap-1 mt-1">
                            {[5, 10, 20, 50, 100].map((lev) => (
                                <button key={lev}
                                    className={`flex-1 py-1 text-xs rounded ${settings.leverage === lev ? 'bg-binanceYellow text-black' : 'bg-panelAlt text-textSecondary hover:bg-border'}`}
                                    onClick={() => setSettings((s) => ({ ...s, leverage: lev }))}
                                >{lev}x</button>
                            ))}
                        </div>
                    </div>

                    {/* Max Orders */}
                    <div>
                        <label className="text-xs text-textSecondary block mb-1">Max Open Orders</label>
                        <input type="number" min="1" max="100" value={settings.max_orders_at_a_time}
                            onChange={(e) => setSettings((s) => ({ ...s, max_orders_at_a_time: parseInt(e.target.value) || 1 }))}
                            className="input-binance w-full" />
                    </div>

                    {/* Trade Size Type — switchable */}
                    <div>
                        <label className="text-xs text-textSecondary block mb-1">Trade Size Type</label>
                        <div className="flex gap-1">
                            <button
                                className={`flex-1 py-2 text-xs rounded ${settings.trade_size_type === 'fixed_usdt' ? 'bg-binanceYellow text-black' : 'bg-panelAlt text-textSecondary'}`}
                                onClick={() => setSettings((s) => ({ ...s, trade_size_type: 'fixed_usdt' }))}
                            >
                                Fixed USDT
                            </button>
                            <button
                                className={`flex-1 py-2 text-xs rounded ${settings.trade_size_type === 'percent_depo' ? 'bg-binanceYellow text-black' : 'bg-panelAlt text-textSecondary'}`}
                                onClick={() => setSettings((s) => ({ ...s, trade_size_type: 'percent_depo' }))}
                            >
                                % of Balance
                            </button>
                        </div>
                    </div>

                    {/* Trade Size Value — depends on type */}
                    <div>
                        <label className="text-xs text-textSecondary block mb-1">
                            {settings.trade_size_type === 'fixed_usdt' ? 'Trade Size (USDT)' : 'Trade Size (% of Overall Balance)'}
                        </label>
                        <div className="relative">
                            <input type="number" min="0.1"
                                max={settings.trade_size_type === 'percent_depo' ? 100 : 100000}
                                step={settings.trade_size_type === 'percent_depo' ? 0.5 : 1}
                                value={settings.trade_size_value}
                                onChange={(e) => setSettings((s) => ({ ...s, trade_size_value: parseFloat(e.target.value) || 0 }))}
                                className="input-binance w-full" />
                            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-textSecondary">
                                {settings.trade_size_type === 'fixed_usdt' ? 'USDT' : '%'}
                            </span>
                        </div>
                    </div>

                    {/* Strategy Type — switchable */}
                    <div>
                        <label className="text-xs text-textSecondary block mb-1">Strategy</label>
                        <div className="flex gap-1">
                            <button
                                className={`flex-1 py-2 text-xs rounded ${settings.strategy_type === 'ml_super_entry' ? 'bg-binanceYellow text-black' : 'bg-panelAlt text-textSecondary'}`}
                                onClick={() => setSettings((s) => ({ ...s, strategy_type: 'ml_super_entry' }))}
                            >
                                ML Super Entry
                            </button>
                            <button
                                className={`flex-1 py-2 text-xs rounded ${settings.strategy_type === 'level_strategy' ? 'bg-binanceYellow text-black' : 'bg-panelAlt text-textSecondary'}`}
                                onClick={() => setSettings((s) => ({ ...s, strategy_type: 'level_strategy' }))}
                            >
                                Level Strategy
                            </button>
                        </div>
                    </div>

                    {/* Order Type */}
                    <div>
                        <label className="text-xs text-textSecondary block mb-1">Order Type</label>
                        <div className="input-binance w-full text-sm">Futures OCO (TP + SL)</div>
                    </div>
                </div>

                <div className="flex gap-2 mt-6">
                    <button className="flex-1 btn-binance btn-binance-secondary" onClick={() => setTradingOptionsModalOpen(false)}>Cancel</button>
                    <button className="flex-1 btn-binance btn-binance-primary" onClick={() => saveMutation.mutate(settings)}>
                        {saveMutation.isPending ? 'Saving...' : 'Save'}
                    </button>
                </div>
            </div>
        </div>
    );
}
