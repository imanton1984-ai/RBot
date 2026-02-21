import { useState, useEffect } from 'react';
import { useTradingStore, useDataStore } from '../store';
import { apiService } from '../api';
import { useQuery } from '@tanstack/react-query';
import { AlertTriangle, Play, Square } from 'lucide-react';

export default function BottomBar() {
  const { tradingMode, setTradingMode } = useTradingStore();
  const [isAutoRunning, setIsAutoRunning] = useState(false);
  const [emergencyLoading, setEmergencyLoading] = useState(false);

  // Load auto trading state
  const { data: tradingState } = useQuery({
    queryKey: ['trading-state'],
    queryFn: () => apiService.getAutoTradingState(),
    refetchInterval: 5000,
  });

  useEffect(() => {
    if (tradingState) {
      setIsAutoRunning(tradingState.is_running);
    }
  }, [tradingState]);

  const handleEmergencyStop = async () => {
    if (confirm('⚠️ EMERGENCY STOP: This will close ALL open positions and stop trading. Are you sure?')) {
      setEmergencyLoading(true);
      try {
        await apiService.emergencyStop();
        setIsAutoRunning(false);
      } catch (error) {
        console.error('Emergency stop failed:', error);
      } finally {
        setEmergencyLoading(false);
      }
    }
  };

  const addLog = useDataStore((s) => s.addTerminalLog);
  const balance = useDataStore((s) => s.balance);
  const connections = useDataStore((s) => s.connections);

  const handleToggleTrading = async () => {
    try {
      if (isAutoRunning) {
        const result = await apiService.stopTrading();
        setIsAutoRunning(result.is_running);
        addLog('info', 'Auto trading stopped. Open positions remain active.');
      } else {
        // Pre-flight checks
        if (balance && balance.overall <= 0) {
          addLog('error', 'Cannot start trading: balance is $0.00. Deposit funds first.');
          return;
        }
        if (balance && balance.available <= 0) {
          addLog('warn', 'Warning: available balance is $0.00. No free margin for new orders.');
        }
        if (connections && !connections.account) {
          addLog('error', 'Cannot start trading: Binance account not connected. Check API keys.');
          return;
        }
        if (connections && !connections.database) {
          addLog('error', 'Cannot start trading: database not connected.');
          return;
        }
        if (connections && !connections.rest_api) {
          addLog('warn', 'Warning: Binance REST API unreachable. Check network/VPN.');
        }

        const result = await apiService.startTrading();
        setIsAutoRunning(result.is_running);
        addLog('info', `Auto trading started. Mode: ${result.trading_mode}`);
      }
    } catch (error: any) {
      addLog('error', `Trading toggle failed: ${error?.message || error}`);
    }
  };

  return (
    <div className="h-14 bg-panel border-t border-border flex items-center px-4 gap-4 shrink-0">
      <div className="flex items-center gap-2">
        {/* Emergency Stop */}
        <button
          className="bg-bear hover:bg-opacity-90 text-white px-6 py-2 rounded-lg font-medium flex items-center gap-2 transition-all"
          onClick={handleEmergencyStop}
          disabled={emergencyLoading}
        >
          <AlertTriangle className="w-4 h-4" />
          {emergencyLoading ? 'STOPPING...' : 'EMERGENCY STOP'}
        </button>

        {/* START / STOP TRADING toggle button */}
        <button
          className={`px-6 py-2 rounded-lg font-medium flex items-center gap-2 transition-all ${isAutoRunning
            ? 'bg-bear hover:bg-opacity-90 text-white'
            : 'bg-bull hover:bg-opacity-90 text-white'
            }`}
          onClick={handleToggleTrading}
        >
          {isAutoRunning ? (
            <>
              <Square className="w-4 h-4" />
              STOP AUTO TRADING
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              START TRADING
            </>
          )}
        </button>
      </div>

      <div className="flex-1" />

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-sm">
          <span className={tradingMode === 'manual' ? 'text-binanceYellow' : 'text-textSecondary'}>
            Manual
          </span>
          <button
            className={`w-10 h-5 rounded-full relative transition-colors ${tradingMode === 'auto' ? 'bg-binanceYellow' : 'bg-panelAlt'
              }`}
            onClick={() => setTradingMode(tradingMode === 'manual' ? 'auto' : 'manual')}
          >
            <div
              className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${tradingMode === 'auto' ? 'translate-x-5' : 'translate-x-0.5'
                }`}
            />
          </button>
          <span className={tradingMode === 'auto' ? 'text-binanceYellow' : 'text-textSecondary'}>
            Auto
          </span>
        </div>

        <div className="w-px h-6 bg-border" />

        <div className="flex items-center gap-2 text-xs text-textSecondary">
          <div className={`w-2 h-2 rounded-full ${isAutoRunning ? 'bg-bull animate-pulse' : 'bg-bear'}`} />
          <span>Auto Trading: {isAutoRunning ? 'ON' : 'OFF'}</span>
        </div>
      </div>
    </div>
  );
}
