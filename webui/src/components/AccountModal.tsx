import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../api';
import { useUiStore } from '../store';
import {
    X, Wallet, ArrowRightLeft, QrCode, Copy, Check,
    ArrowDown, ArrowUp, Loader2, AlertCircle,
} from 'lucide-react';
import type { TransferRequest } from '../types';

const NETWORKS = [
    { id: 'evm', label: 'ERC-20 / BSC / ARB', address: '0x34dce478d39b8fb06498f26ec7ca6169451c2acb', color: '#627EEA' },
    { id: 'tron', label: 'TRON (TRC-20)', address: 'TN7GnuWDJnqZ873KAtL9JXS2C6bx9BVZ7g', color: '#FF0013' },
] as const;

export default function AccountModal() {
    const { walletModalOpen, setWalletModalOpen } = useUiStore();
    const [activeTab, setActiveTab] = useState<'balance' | 'deposit' | 'transfer'>('balance');
    const [copied, setCopied] = useState(false);
    const [selectedNetwork, setSelectedNetwork] = useState<'evm' | 'tron'>('evm');

    // Transfer state
    const [transferDirection, setTransferDirection] = useState<'SPOT_TO_FUTURES' | 'FUTURES_TO_SPOT'>('SPOT_TO_FUTURES');
    const [transferAmount, setTransferAmount] = useState('');
    const [transferResult, setTransferResult] = useState<{ success: boolean; message: string } | null>(null);

    const queryClient = useQueryClient();

    const { data: balances, isLoading } = useQuery({
        queryKey: ['account-balances'],
        queryFn: () => apiService.getAccountBalances(),
        enabled: walletModalOpen,
        refetchInterval: 10000,
    });

    const transferMutation = useMutation({
        mutationFn: (req: TransferRequest) => apiService.transferBetweenAccounts(req),
        onSuccess: (data) => {
            setTransferResult({ success: data.success, message: data.message });
            if (data.success) {
                setTransferAmount('');
                queryClient.invalidateQueries({ queryKey: ['account-balances'] });
                queryClient.invalidateQueries({ queryKey: ['balance'] });
            }
        },
        onError: (err: any) => {
            setTransferResult({ success: false, message: err?.message || 'Transfer failed' });
        },
    });

    if (!walletModalOpen) return null;

    const currentNetwork = NETWORKS.find(n => n.id === selectedNetwork)!;
    const depositAddress = currentNetwork.address;

    const handleCopy = () => {
        navigator.clipboard.writeText(depositAddress);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    const handleTransfer = () => {
        const amount = parseFloat(transferAmount);
        if (isNaN(amount) || amount <= 0) {
            setTransferResult({ success: false, message: 'Enter a valid amount' });
            return;
        }
        setTransferResult(null);
        transferMutation.mutate({ direction: transferDirection, amount });
    };

    const handleMaxTransfer = () => {
        if (!balances) return;
        const max = transferDirection === 'SPOT_TO_FUTURES'
            ? balances.spot_available
            : balances.futures_available;
        setTransferAmount(max.toFixed(2));
    };

    // QR code via external API
    const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(depositAddress)}`;

    return (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-8">
            <div className="bg-panel rounded-2xl w-full max-w-lg overflow-hidden flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-border">
                    <div className="flex items-center gap-3">
                        <Wallet className="w-7 h-7 text-binanceYellow" />
                        <h2 className="text-xl font-bold text-textPrimary">Account Balance</h2>
                    </div>
                    <button
                        onClick={() => setWalletModalOpen(false)}
                        className="p-2 hover:bg-panelAlt rounded-lg transition-colors"
                    >
                        <X className="w-5 h-5 text-textSecondary" />
                    </button>
                </div>

                {/* Tab buttons */}
                <div className="flex items-center gap-1 p-4 border-b border-border">
                    {(['balance', 'deposit', 'transfer'] as const).map((tab) => (
                        <button
                            key={tab}
                            onClick={() => { setActiveTab(tab); setTransferResult(null); }}
                            className={`flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2 ${activeTab === tab
                                ? 'bg-binanceYellow text-black'
                                : 'bg-panelAlt text-textSecondary hover:text-textPrimary'
                                }`}
                        >
                            {tab === 'balance' && <Wallet className="w-4 h-4" />}
                            {tab === 'deposit' && <QrCode className="w-4 h-4" />}
                            {tab === 'transfer' && <ArrowRightLeft className="w-4 h-4" />}
                            {tab.charAt(0).toUpperCase() + tab.slice(1)}
                        </button>
                    ))}
                </div>

                {/* Content */}
                <div className="p-6">
                    {isLoading ? (
                        <div className="flex items-center justify-center h-40">
                            <Loader2 className="w-8 h-8 text-binanceYellow animate-spin" />
                        </div>
                    ) : (
                        <>
                            {/* ─── Balance Tab ──────────────────────────────── */}
                            {activeTab === 'balance' && (
                                <div className="space-y-4">
                                    {/* Total */}
                                    <div className="bg-panelAlt rounded-xl p-5 text-center">
                                        <div className="text-textSecondary text-xs mb-1">Total USDT Balance</div>
                                        <div className="text-3xl font-bold text-binanceYellow">
                                            ${balances?.total_usdt?.toFixed(2) ?? '0.00'}
                                        </div>
                                    </div>

                                    {/* Spot */}
                                    <div className="bg-panelAlt rounded-xl p-4">
                                        <div className="flex items-center gap-2 mb-3">
                                            <div className="w-2 h-2 rounded-full bg-blue-400" />
                                            <span className="text-sm font-medium text-textPrimary">Spot Account</span>
                                        </div>
                                        <div className="space-y-2">
                                            <div className="flex justify-between text-xs">
                                                <span className="text-textSecondary">Total USDT</span>
                                                <span className="text-textPrimary font-medium">
                                                    ${balances?.spot_usdt?.toFixed(2) ?? '0.00'}
                                                </span>
                                            </div>
                                            <div className="flex justify-between text-xs">
                                                <span className="text-textSecondary">Available</span>
                                                <span className="text-textPrimary">
                                                    ${balances?.spot_available?.toFixed(2) ?? '0.00'}
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Futures */}
                                    <div className="bg-panelAlt rounded-xl p-4">
                                        <div className="flex items-center gap-2 mb-3">
                                            <div className="w-2 h-2 rounded-full bg-binanceYellow" />
                                            <span className="text-sm font-medium text-textPrimary">Futures Account</span>
                                        </div>
                                        <div className="space-y-2">
                                            <div className="flex justify-between text-xs">
                                                <span className="text-textSecondary">Wallet Balance</span>
                                                <span className="text-textPrimary font-medium">
                                                    ${balances?.futures_usdt?.toFixed(2) ?? '0.00'}
                                                </span>
                                            </div>
                                            <div className="flex justify-between text-xs">
                                                <span className="text-textSecondary">Available</span>
                                                <span className="text-textPrimary">
                                                    ${balances?.futures_available?.toFixed(2) ?? '0.00'}
                                                </span>
                                            </div>
                                            <div className="flex justify-between text-xs">
                                                <span className="text-textSecondary">Unrealized PnL</span>
                                                <span className={
                                                    (balances?.futures_unrealized_pnl ?? 0) >= 0 ? 'text-bull' : 'text-bear'
                                                }>
                                                    {(balances?.futures_unrealized_pnl ?? 0) >= 0 ? '+' : ''}
                                                    ${balances?.futures_unrealized_pnl?.toFixed(2) ?? '0.00'}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* ─── Deposit Tab ──────────────────────────────── */}
                            {activeTab === 'deposit' && (
                                <div className="space-y-5">
                                    <div className="text-center">
                                        <div className="text-sm text-textSecondary mb-1">Deposit USDT</div>
                                        <div className="text-xs text-binanceYellow">Select network and send USDT to the address</div>
                                    </div>

                                    {/* Network Selector */}
                                    <div className="flex items-center gap-2">
                                        {NETWORKS.map((net) => (
                                            <button
                                                key={net.id}
                                                onClick={() => { setSelectedNetwork(net.id); setCopied(false); }}
                                                className={`flex-1 px-3 py-2.5 rounded-lg text-xs font-medium transition-colors flex items-center justify-center gap-2 border ${selectedNetwork === net.id
                                                        ? 'border-binanceYellow bg-binanceYellow/10 text-textPrimary'
                                                        : 'border-border bg-panelAlt text-textSecondary hover:text-textPrimary'
                                                    }`}
                                            >
                                                <div
                                                    className="w-2.5 h-2.5 rounded-full shrink-0"
                                                    style={{ backgroundColor: net.color }}
                                                />
                                                {net.label}
                                            </button>
                                        ))}
                                    </div>

                                    {/* QR Code */}
                                    <div className="flex justify-center">
                                        <div className="bg-white rounded-xl p-3">
                                            <img
                                                src={qrUrl}
                                                alt="USDT Deposit QR Code"
                                                className="w-48 h-48"
                                                loading="eager"
                                            />
                                        </div>
                                    </div>

                                    {/* Address */}
                                    <div className="bg-panelAlt rounded-xl p-4">
                                        <div className="text-xs text-textSecondary mb-2">
                                            {currentNetwork.label} — Wallet Address
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <code className="flex-1 text-xs text-textPrimary bg-panel px-3 py-2 rounded-lg break-all font-mono">
                                                {depositAddress}
                                            </code>
                                            <button
                                                onClick={handleCopy}
                                                className="p-2 hover:bg-panel rounded-lg transition-colors shrink-0"
                                                title="Copy address"
                                            >
                                                {copied ? (
                                                    <Check className="w-4 h-4 text-bull" />
                                                ) : (
                                                    <Copy className="w-4 h-4 text-textSecondary" />
                                                )}
                                            </button>
                                        </div>
                                    </div>

                                    <div className="bg-panelAlt/50 rounded-lg p-3 flex items-start gap-2">
                                        <AlertCircle className="w-4 h-4 text-binanceYellow shrink-0 mt-0.5" />
                                        <span className="text-[11px] text-textSecondary leading-relaxed">
                                            Only send USDT on the <strong>{currentNetwork.label}</strong> network to this address.
                                            Sending tokens on the wrong network may result in permanent loss.
                                        </span>
                                    </div>
                                </div>
                            )}

                            {/* ─── Transfer Tab ─────────────────────────────── */}
                            {activeTab === 'transfer' && (
                                <div className="space-y-5">
                                    {/* Direction selector */}
                                    <div className="bg-panelAlt rounded-xl p-4">
                                        <div className="flex items-center gap-3">
                                            {/* From */}
                                            <div className="flex-1 bg-panel rounded-lg p-3 text-center">
                                                <div className="text-[10px] text-textSecondary mb-1">From</div>
                                                <div className="text-sm font-medium text-textPrimary">
                                                    {transferDirection === 'SPOT_TO_FUTURES' ? 'Spot' : 'Futures'}
                                                </div>
                                                <div className="text-[10px] text-textSecondary mt-1">
                                                    ${transferDirection === 'SPOT_TO_FUTURES'
                                                        ? (balances?.spot_available?.toFixed(2) ?? '0.00')
                                                        : (balances?.futures_available?.toFixed(2) ?? '0.00')
                                                    }
                                                </div>
                                            </div>

                                            {/* Swap button */}
                                            <button
                                                onClick={() =>
                                                    setTransferDirection(
                                                        transferDirection === 'SPOT_TO_FUTURES'
                                                            ? 'FUTURES_TO_SPOT'
                                                            : 'SPOT_TO_FUTURES'
                                                    )
                                                }
                                                className="p-2 bg-binanceYellow rounded-full hover:bg-binanceYellow/80 transition-colors"
                                                title="Swap direction"
                                            >
                                                <ArrowRightLeft className="w-4 h-4 text-black" />
                                            </button>

                                            {/* To */}
                                            <div className="flex-1 bg-panel rounded-lg p-3 text-center">
                                                <div className="text-[10px] text-textSecondary mb-1">To</div>
                                                <div className="text-sm font-medium text-textPrimary">
                                                    {transferDirection === 'SPOT_TO_FUTURES' ? 'Futures' : 'Spot'}
                                                </div>
                                                <div className="text-[10px] text-textSecondary mt-1">
                                                    ${transferDirection === 'SPOT_TO_FUTURES'
                                                        ? (balances?.futures_available?.toFixed(2) ?? '0.00')
                                                        : (balances?.spot_available?.toFixed(2) ?? '0.00')
                                                    }
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Direction Label */}
                                    <div className="flex items-center justify-center gap-2 text-xs text-textSecondary">
                                        {transferDirection === 'SPOT_TO_FUTURES' ? (
                                            <>
                                                <span>Spot</span>
                                                <ArrowDown className="w-3 h-3 text-binanceYellow" />
                                                <span>Futures</span>
                                            </>
                                        ) : (
                                            <>
                                                <span>Futures</span>
                                                <ArrowUp className="w-3 h-3 text-binanceYellow" />
                                                <span>Spot</span>
                                            </>
                                        )}
                                    </div>

                                    {/* Amount input */}
                                    <div className="bg-panelAlt rounded-xl p-4">
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="text-xs text-textSecondary">Amount (USDT)</span>
                                            <button
                                                onClick={handleMaxTransfer}
                                                className="text-[10px] text-binanceYellow hover:underline"
                                            >
                                                MAX
                                            </button>
                                        </div>
                                        <input
                                            type="number"
                                            value={transferAmount}
                                            onChange={(e) => setTransferAmount(e.target.value)}
                                            placeholder="0.00"
                                            className="w-full bg-panel border border-border rounded-lg px-3 py-2.5 text-sm text-textPrimary 
                                 focus:outline-none focus:border-binanceYellow transition-colors"
                                            min="0"
                                            step="0.01"
                                        />
                                    </div>

                                    {/* Transfer button */}
                                    <button
                                        onClick={handleTransfer}
                                        disabled={transferMutation.isPending || !transferAmount}
                                        className="w-full bg-binanceYellow text-black font-semibold py-3 rounded-xl 
                               hover:bg-binanceYellow/90 disabled:opacity-50 disabled:cursor-not-allowed
                               transition-colors flex items-center justify-center gap-2"
                                    >
                                        {transferMutation.isPending ? (
                                            <>
                                                <Loader2 className="w-4 h-4 animate-spin" />
                                                Processing...
                                            </>
                                        ) : (
                                            <>
                                                <ArrowRightLeft className="w-4 h-4" />
                                                Transfer
                                            </>
                                        )}
                                    </button>

                                    {/* Result message */}
                                    {transferResult && (
                                        <div
                                            className={`rounded-lg p-3 text-sm flex items-start gap-2 ${transferResult.success
                                                ? 'bg-bull/10 text-bull'
                                                : 'bg-bear/10 text-bear'
                                                }`}
                                        >
                                            {transferResult.success ? (
                                                <Check className="w-4 h-4 shrink-0 mt-0.5" />
                                            ) : (
                                                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                                            )}
                                            <span className="text-xs">{transferResult.message}</span>
                                        </div>
                                    )}
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
