import { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import { useTradingStore, useUiStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { ChevronDown, LineChart, Table, Bell, Settings2 } from 'lucide-react';

const TIMEFRAMES = [
  { label: '1m', value: 1 },
  { label: '5m', value: 5 },
  { label: '15m', value: 15 },
  { label: '1h', value: 60 },
  { label: '4h', value: 240 },
  { label: '1d', value: 1440 },
];

export default function ChartPanel() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const { currentPair, currentTf, setCurrentTf } = useTradingStore();
  const { chartView, setChartView, setSignalsModalOpen } = useUiStore();
  const [indicatorsOpen, setIndicatorsOpen] = useState(false);

  // Load candles only for active pair/TF - refetch every 15 seconds
  const { data: candleData, isLoading } = useQuery({
    queryKey: ['candles', currentPair, currentTf],
    queryFn: () => apiService.getCandles(currentPair, currentTf, 1000),
    refetchInterval: 15000, // 15 seconds
    staleTime: 10000,
    retry: 1,
  });

  const [chartInitialized, setChartInitialized] = useState(false);

  // Initialize chart when data arrives
  useEffect(() => {
    if (!chartContainerRef.current || !candleData || candleData.length === 0 || chartInitialized) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: '#0B0F14' },
        textColor: '#9CA3AF',
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.03)' },
        horzLines: { color: 'rgba(255,255,255,0.03)' },
      },
      crosshair: { mode: 1 },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.06)',
      },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#0ECB81',
      downColor: '#F6465D',
      borderUpColor: '#0ECB81',
      borderDownColor: '#F6465D',
      wickUpColor: '#0ECB81',
      wickDownColor: '#F6465D',
    });

    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    });

    const candles = candleData.map((c: any) => ({
      time: (c.t / 1000) as any,
      open: c.o,
      high: c.h,
      low: c.l,
      close: c.c,
    }));

    const volume = candleData.map((c: any) => ({
      time: (c.t / 1000) as any,
      value: c.v,
      color: c.c >= c.o ? 'rgba(14, 203, 129, 0.5)' : 'rgba(246, 70, 93, 0.5)',
    }));

    candlestickSeries.setData(candles);
    volumeSeries.setData(volume);

    setChartInitialized(true);

    return () => { chart.remove(); setChartInitialized(false); };
  }, [currentPair, currentTf]); // Re-init only when pair/TF changes

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="h-12 border-b border-border flex items-center px-3 gap-2 shrink-0">
        <button
          className={`px-3 py-1.5 rounded text-sm flex items-center gap-1 ${
            chartView === 'chart' ? 'bg-binanceYellow text-black' : 'text-textSecondary hover:bg-panelAlt'
          }`}
          onClick={() => setChartView('chart')}
        >
          <LineChart className="w-3 h-3" /> Chart
        </button>
        <button
          className={`px-3 py-1.5 rounded text-sm flex items-center gap-1 ${
            chartView === 'positions' ? 'bg-binanceYellow text-black' : 'text-textSecondary hover:bg-panelAlt'
          }`}
          onClick={() => setChartView('positions')}
        >
          <Table className="w-3 h-3" /> Positions
        </button>

        <div className="w-px h-6 bg-border mx-2" />

        <div className="flex items-center gap-1">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf.value}
              className={`px-2 py-1 rounded text-xs ${
                currentTf === tf.value ? 'bg-panelAlt text-textPrimary' : 'text-textSecondary hover:bg-panelAlt'
              }`}
              onClick={() => setCurrentTf(tf.value)}
            >
              {tf.label}
            </button>
          ))}
        </div>

        <div className="w-px h-6 bg-border mx-2" />

        <div className="relative">
          <button
            className="px-3 py-1.5 rounded text-sm text-textSecondary hover:bg-panelAlt flex items-center gap-1"
            onClick={() => setIndicatorsOpen(!indicatorsOpen)}
          >
            <Settings2 className="w-3 h-3" /> Indicators <ChevronDown className="w-3 h-3" />
          </button>
          {indicatorsOpen && (
            <div className="absolute top-full left-0 mt-1 w-40 bg-panel border border-border rounded-lg shadow-xl z-50 p-2">
              <div className="text-xs text-textSecondary">EMA, RSI, MACD</div>
            </div>
          )}
        </div>

        <button
          className="px-3 py-1.5 rounded text-sm text-textSecondary hover:bg-panelAlt flex items-center gap-1"
          onClick={() => setSignalsModalOpen(true)}
        >
          <Bell className="w-3 h-3" /> Signals
        </button>

        <div className="flex-1" />
        
        {isLoading && (
          <div className="text-xs text-textSecondary animate-pulse">Loading...</div>
        )}
      </div>

      {chartView === 'chart' ? (
        <div ref={chartContainerRef} className="flex-1 min-h-0 relative">
          {isLoading && !chartInitialized && (
            <div className="absolute inset-0 flex items-center justify-center text-textSecondary">
              Loading candles...
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-textSecondary">Positions View</div>
      )}
    </div>
  );
}
