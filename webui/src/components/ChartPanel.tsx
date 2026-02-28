import { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi } from 'lightweight-charts';
import { useTradingStore, useUiStore, useDataStore, useWsStore } from '../store';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../api';
import { ChevronDown, LineChart, LayoutGrid, Settings2 } from 'lucide-react';
import type { Position } from '../types';
import { formatPrice } from '../utils/format';

const TIMEFRAMES = [
  { label: '1m', value: 1 },
  { label: '5m', value: 5 },
  { label: '15m', value: 15 },
  { label: '1h', value: 60 },
  { label: '4h', value: 240 },
  { label: '1d', value: 1440 },
];

// ─── Positions View (Grid of open positions) ─────────────────────────
function PositionsGridView() {
  const positions = useDataStore((s) => s.positions);
  const { setCurrentPair } = useTradingStore();
  const setChartView = useUiStore((s) => s.setChartView);

  const { data: openPositions } = useQuery({
    queryKey: ['positions-open'],
    queryFn: () => apiService.getOpenPositions(),
    refetchInterval: 5000,
  });

  const currentPositions = openPositions || positions;
  const gridSlots = Array.from({ length: 10 }, (_, i) => currentPositions[i] || null);

  const handleClick = (pos: Position) => {
    setCurrentPair(pos.pair);
    setChartView('chart');
  };

  return (
    <div className="flex-1 p-3 overflow-y-auto">
      <div className="grid grid-cols-5 grid-rows-2 gap-2 h-full">
        {gridSlots.map((pos, idx) => (
          <div
            key={idx}
            className={`rounded-lg border p-3 flex flex-col justify-between transition-all cursor-pointer ${pos
              ? `border-border bg-panelAlt hover:border-binanceYellow hover:bg-panel ${pos.pnl_usdt >= 0 ? 'hover:shadow-bull/10' : 'hover:shadow-bear/10'} hover:shadow-lg`
              : 'border-border/30 bg-panel/50'
              }`}
            onClick={() => pos && handleClick(pos)}
          >
            {pos ? (
              <>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-textPrimary truncate">{pos.pair}</span>
                  <span className={`text-xs px-1 py-0.5 rounded ${pos.side === 'LONG' ? 'bg-bull/20 text-bull' : 'bg-bear/20 text-bear'}`}>
                    {pos.side}
                  </span>
                </div>
                <div className="space-y-0.5 text-xs">
                  <div className="flex justify-between">
                    <span className="text-textSecondary">Entry</span>
                    <span className="text-textPrimary">${formatPrice(pos.entry_price)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-textSecondary">Current</span>
                    <span className="text-textPrimary">${formatPrice(pos.current_price)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-textSecondary">SL</span>
                    <span className="text-bear">${formatPrice(pos.stop_loss)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-textSecondary">Bars Left</span>
                    <span className="text-binanceYellow">{pos.candles_left}</span>
                  </div>
                </div>
                {/* PnL mini bar */}
                <div className={`mt-2 h-6 rounded flex items-center justify-center text-xs font-bold ${pos.pnl_usdt >= 0 ? 'bg-bull/20 text-bull' : 'bg-bear/20 text-bear'
                  }`}>
                  {pos.pnl_usdt >= 0 ? '+' : ''}${pos.pnl_usdt?.toFixed(2)}
                </div>
              </>
            ) : (
              <div className="flex items-center justify-center h-full text-textSecondary/30 text-xs">
                Empty
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main Chart Panel ────────────────────────────────────────────────
export default function ChartPanel() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const priceLinesRef = useRef<any[]>([]);
  // Track whether initial fitContent has been done for this pair/tf combo
  const initialFitDoneRef = useRef<string>('');

  const queryClient = useQueryClient();
  const { currentPair, currentTf, setCurrentTf } = useTradingStore();
  const chartView = useUiStore((s) => s.chartView);
  const setChartView = useUiStore((s) => s.setChartView);
  const setSignalsModalOpen = useUiStore((s) => s.setSignalsModalOpen);
  const setCandles = useDataStore((s) => s.setCandles);
  const positions = useDataStore((s) => s.positions);
  const [indicatorsOpen, setIndicatorsOpen] = useState(false);
  const [chartReady, setChartReady] = useState(false);

  // Invalidate candle query when timeframe or pair changes to ensure fresh data
  useEffect(() => {
    queryClient.invalidateQueries({ queryKey: ['candles', currentPair, currentTf] });
  }, [currentPair, currentTf, queryClient]);

  // Load candles via HTTP — serves as initial load & periodic full sync.
  // Real-time updates come via WebSocket (candle_update) every 5s from candle_broadcaster.
  // HTTP poll every 30s is a fallback to catch any missed WS messages or closed candles.
  const { data: candleData, isLoading } = useQuery({
    queryKey: ['candles', currentPair, currentTf],
    queryFn: () => apiService.getCandles(currentPair, currentTf, 5000),
    refetchInterval: 30000,
    staleTime: 10000,
    retry: 1,
    refetchOnWindowFocus: true,
  });

  // Load open positions for current pair (for TP/SL lines)
  // Refresh every 2s for smooth PnL and price updates
  const { data: openPositions } = useQuery({
    queryKey: ['positions-open'],
    queryFn: () => apiService.getOpenPositions(),
    refetchInterval: 2000,
    staleTime: 1500,
    retry: 1,
    refetchOnWindowFocus: true,
  });

  // Find position for current pair
  const currentPairPositions = (openPositions || positions).filter(
    (p: Position) => p.pair === currentPair
  );

  // Create chart
  useEffect(() => {
    if (chartView !== 'chart') return;
    if (!chartContainerRef.current) return;

    const container = chartContainerRef.current;
    const w = Math.max(container.clientWidth, 100);
    const h = Math.max(container.clientHeight, 100);

    const chart = createChart(container, {
      width: w,
      height: h,
      autoSize: true,
      layout: {
        background: { color: '#0B0F14' },
        textColor: '#9CA3AF',
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.03)' },
        horzLines: { color: 'rgba(255,255,255,0.03)' },
      },
      crosshair: { mode: 0 },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        timeVisible: true,
        secondsVisible: false,
        // Right offset: ~40px ≈ 1cm+ gap from right edge of chart area
        rightOffset: 12,
        // Don't shift to bar edge automatically
        shiftVisibleRangeOnNewBar: true,
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
      priceScaleId: 'volume',
    });

    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });
    candlestickSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.02, bottom: 0.18 },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candlestickSeries;
    volumeSeriesRef.current = volumeSeries;
    setChartReady(true);

    // Reset initialFitDone when chart is recreated
    initialFitDoneRef.current = '';

    return () => {
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      setChartReady(false);
    };
  }, [chartView]);

  // Update chart data
  useEffect(() => {
    if (!chartReady) return;
    if (!candleData || candleData.length === 0) return;
    if (!candleSeriesRef.current || !volumeSeriesRef.current) return;

    // FIX #8: Dynamic price precision for the right price axis.
    // Determine precision from the latest close price so low-price coins
    // (e.g., DOGE at 0.10234) show enough decimals instead of "0.10".
    const lastClose = candleData[candleData.length - 1]?.c || 0;
    let precision = 2;
    let minMove = 0.01;
    if (lastClose > 0 && lastClose < 1) { precision = 5; minMove = 0.00001; }
    if (lastClose > 0 && lastClose < 0.1) { precision = 6; minMove = 0.000001; }
    if (lastClose > 0 && lastClose < 0.01) { precision = 7; minMove = 0.0000001; }
    if (lastClose > 0 && lastClose < 0.001) { precision = 8; minMove = 0.00000001; }

    candleSeriesRef.current.applyOptions({
      priceFormat: {
        type: 'price',
        precision,
        minMove,
      },
    });

    const candles = candleData.map((c: any) => ({
      time: Math.floor(c.t / 1000) as any,
      open: c.o,
      high: c.h,
      low: c.l,
      close: c.c,
    }));

    const volume = candleData.map((c: any) => ({
      time: Math.floor(c.t / 1000) as any,
      value: c.v,
      color: c.c >= c.o ? 'rgba(14, 203, 129, 0.3)' : 'rgba(246, 70, 93, 0.3)',
    }));

    // Whitespace bars for right-side padding (rightOffset handles visual gap)
    const whitespaceBars = 3;

    const lastCandle = candleData[candleData.length - 1];
    const lastTimeSec = Math.floor(lastCandle.t / 1000);
    const tfSec = currentTf * 60;
    const whitespace: any[] = [];
    for (let i = 1; i <= whitespaceBars; i++) {
      whitespace.push({ time: (lastTimeSec + i * tfSec) as any });
    }

    candleSeriesRef.current.setData([...candles, ...whitespace]);
    volumeSeriesRef.current.setData(volume);
    setCandles(candleData);

    // Only fitContent on FIRST load for this pair/tf combo.
    // On subsequent updates (new candles), just let the chart scroll naturally
    // via shiftVisibleRangeOnNewBar. This prevents annoying auto-zoom.
    const fitKey = `${currentPair}:${currentTf}`;
    if (initialFitDoneRef.current !== fitKey) {
      chartRef.current?.timeScale().fitContent();
      initialFitDoneRef.current = fitKey;
    } else {
      // Scroll to show the latest candle without changing zoom level
      chartRef.current?.timeScale().scrollToRealTime();
    }
  }, [candleData, chartReady, currentPair, currentTf, setCandles]);

  // Draw TP/SL lines and Candles Left marker for active positions
  useEffect(() => {
    if (!chartReady || !candleSeriesRef.current) return;

    const series = candleSeriesRef.current;

    // 1. Remove ALL old price lines before creating new ones
    for (const line of priceLinesRef.current) {
      try { series.removePriceLine(line); } catch (_) { /* ignore */ }
    }
    priceLinesRef.current = [];

    // 2. Clear old markers
    series.setMarkers([]);

    // 3. Add TP/SL lines for each position on this pair
    currentPairPositions.forEach((pos: Position) => {
      if (pos.take_profit > 0) {
        const line = series.createPriceLine({
          price: pos.take_profit,
          color: '#0ECB81',
          lineWidth: 2,
          lineStyle: 0,
          axisLabelVisible: true,
          title: `TP ${pos.side}`,
        });
        priceLinesRef.current.push(line);
      }

      if (pos.stop_loss > 0) {
        const line = series.createPriceLine({
          price: pos.stop_loss,
          color: '#F6465D',
          lineWidth: 2,
          lineStyle: 0,
          axisLabelVisible: true,
          title: `SL ${pos.side}`,
        });
        priceLinesRef.current.push(line);
      }
    });

    // FIX #10: Orange circle placed on the bar where the order was OPENED (not in the future).
    // FIX #9: Show real bars_left countdown value in the marker text.
    // Size reduced by 25% from 3 → 2.
    if (candleData && candleData.length > 0 && currentPairPositions.length > 0) {
      const markers: any[] = [];
      const chartTfSeconds = currentTf * 60;

      currentPairPositions.forEach((pos: Position) => {
        // FIX #10: Place marker on the bar where the order was opened
        if (pos.open_time) {
          const openTimeSec = Math.floor(new Date(pos.open_time).getTime() / 1000);
          // Snap to nearest chart-timeframe bar boundary
          const snappedTime = Math.floor(openTimeSec / chartTfSeconds) * chartTfSeconds;

          // FIX #9: Show bars_left countdown value
          const posTfMinutes = pos.tf_minutes || currentTf;
          const barsLeftText = pos.candles_left > 0
            ? `📊 ${pos.candles_left} bars (${posTfMinutes}m)`
            : '⏰ Expired';

          markers.push({
            time: snappedTime as any,
            position: 'inBar',
            color: '#F0B90B',
            shape: 'circle',
            size: 2,  // FIX #10: reduced by 25% from 3 → 2
            text: barsLeftText,
          });
        }
      });

      if (markers.length > 0) {
        markers.sort((a, b) => (a.time as number) - (b.time as number));
        series.setMarkers(markers);
      }
    }

    // Cleanup: remove lines when effect re-runs or unmounts
    return () => {
      if (candleSeriesRef.current) {
        for (const line of priceLinesRef.current) {
          try { candleSeriesRef.current.removePriceLine(line); } catch (_) { /* ignore */ }
        }
        priceLinesRef.current = [];
        try { candleSeriesRef.current.setMarkers([]); } catch (_) { /* ignore */ }
      }
    };
  }, [chartReady, currentPairPositions, candleData, currentTf]);

  // ─── WebSocket real-time candle update (every ~5s from candle_broadcaster) ───
  // Subscribes to lastCandleUpdate from the store. When a candle_update arrives
  // matching the current pair+tf, directly updates the chart via .update()
  // for smooth, near-real-time rendering without full HTTP refetch.
  //
  // PERF: Wrapped in try/catch to prevent lightweight-charts errors from
  // crashing the entire React component tree (causes blank screen).
  // Validates data before passing to chart to reject stale/invalid candles.
  const lastCandleUpdate = useDataStore((s) => s.lastCandleUpdate);

  useEffect(() => {
    if (!chartReady || !candleSeriesRef.current || !volumeSeriesRef.current) return;
    if (!lastCandleUpdate) return;

    // Only process updates for the currently viewed pair+tf
    if (lastCandleUpdate.pair !== currentPair || lastCandleUpdate.tf !== currentTf) return;

    // Validate candle data: reject zero/NaN/undefined values
    const { t, o, h, l, c, v } = lastCandleUpdate;
    if (!t || !o || !c || t <= 0 || isNaN(o) || isNaN(c)) return;

    const timeSec = Math.floor(t / 1000) as any;

    // Reject timestamps in the distant past (before 2020) — stale data
    if (timeSec < 1577836800) return; // 2020-01-01

    try {
      // Update candlestick series
      candleSeriesRef.current.update({
        time: timeSec,
        open: o,
        high: h || Math.max(o, c),
        low: l || Math.min(o, c),
        close: c,
      });

      // Update volume series
      volumeSeriesRef.current.update({
        time: timeSec,
        value: v || 0,
        color: c >= o
          ? 'rgba(14, 203, 129, 0.3)'
          : 'rgba(246, 70, 93, 0.3)',
      });
    } catch (err) {
      // lightweight-charts can throw on invalid time sequences or data.
      // Log but don't crash — the next HTTP poll will re-sync the full chart.
      console.warn('[ChartPanel] candle update error (non-fatal):', err);
    }
  }, [lastCandleUpdate, chartReady, currentPair, currentTf]);

  return (
    <div className="h-full w-full flex flex-col overflow-hidden">
      <div className="h-12 border-b border-border flex items-center px-3 gap-2 shrink-0">
        <button
          className={`px-3 py-1.5 rounded text-sm flex items-center gap-1 ${chartView === 'chart' ? 'bg-binanceYellow text-black' : 'text-textSecondary hover:bg-panelAlt'
            }`}
          onClick={() => setChartView('chart')}
        >
          <LineChart className="w-3 h-3" /> Chart
        </button>
        <button
          className={`px-3 py-1.5 rounded text-sm flex items-center gap-1 ${chartView === 'positions' ? 'bg-binanceYellow text-black' : 'text-textSecondary hover:bg-panelAlt'
            }`}
          onClick={() => setChartView('positions')}
        >
          <LayoutGrid className="w-3 h-3" /> Positions
        </button>

        <div className="w-px h-6 bg-border mx-2" />

        <div className="flex items-center gap-1">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf.value}
              className={`px-2 py-1 rounded text-xs ${currentTf === tf.value ? 'bg-panelAlt text-textPrimary' : 'text-textSecondary hover:bg-panelAlt'
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

        <div className="flex-1" />

        {/* Position count badge */}
        {currentPairPositions.length > 0 && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-binanceYellow/10 border border-binanceYellow/30">
            <span className="text-xs text-binanceYellow font-medium">
              {currentPairPositions.length} open
            </span>
          </div>
        )}

        {isLoading && (
          <div className="text-xs text-textSecondary animate-pulse">Loading...</div>
        )}
      </div>

      {chartView === 'chart' ? (
        <div ref={chartContainerRef} className="flex-1 min-h-0 relative">
          {isLoading && !candleData && (
            <div className="absolute inset-0 flex items-center justify-center text-textSecondary">
              Loading candles...
            </div>
          )}
        </div>
      ) : (
        <PositionsGridView />
      )}
    </div>
  );
}
