import { useTradingStore } from '../store';
import { apiService } from '../api';
import { AlertTriangle, RefreshCw, Play } from 'lucide-react';

export default function BottomBar() {
  const { tradingMode, setTradingMode } = useTradingStore();

  const handleEmergencyStop = async () => {
    if (confirm('Are you sure you want to trigger EMERGENCY STOP?')) {
      await apiService.emergencyStop();
    }
  };

  const handleReload = async () => {
    await apiService.reloadBase();
  };

  const handleStart = async () => {
    await apiService.startTrading();
  };

  return (
    <div className="h-14 bg-panel border-t border-border flex items-center px-4 gap-4 shrink-0">
      <div className="flex items-center gap-2">
        <button
          className="bg-bear hover:bg-opacity-90 text-white px-6 py-2 rounded-lg font-medium flex items-center gap-2 transition-all"
          onClick={handleEmergencyStop}
        >
          <AlertTriangle className="w-4 h-4" />
          EMERGENCY STOP
        </button>
        <button
          className="bg-binanceYellow hover:bg-opacity-90 text-black px-6 py-2 rounded-lg font-medium flex items-center gap-2 transition-all"
          onClick={handleReload}
        >
          <RefreshCw className="w-4 h-4" />
          RELOAD BASE
        </button>
        <button
          className="bg-bull hover:bg-opacity-90 text-white px-6 py-2 rounded-lg font-medium flex items-center gap-2 transition-all"
          onClick={handleStart}
        >
          <Play className="w-4 h-4" />
          START TRADING
        </button>
      </div>

      <div className="flex-1" />

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-sm">
          <span className={tradingMode === 'manual' ? 'text-binanceYellow' : 'text-textSecondary'}>
            Manual
          </span>
          <button
            className={`w-10 h-5 rounded-full relative transition-colors ${
              tradingMode === 'auto' ? 'bg-binanceYellow' : 'bg-panelAlt'
            }`}
            onClick={() => setTradingMode(tradingMode === 'manual' ? 'auto' : 'manual')}
          >
            <div
              className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                tradingMode === 'auto' ? 'translate-x-5' : 'translate-x-0.5'
              }`}
            />
          </button>
          <span className={tradingMode === 'auto' ? 'text-binanceYellow' : 'text-textSecondary'}>
            Auto
          </span>
        </div>

        <div className="w-px h-6 bg-border" />

        <div className="flex items-center gap-2 text-xs text-textSecondary">
          <div className="w-2 h-2 rounded-full bg-bull" />
          <span>Trading: {tradingMode === 'auto' ? 'ON' : 'OFF'}</span>
        </div>
      </div>
    </div>
  );
}
