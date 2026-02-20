import { useEffect, useRef, useCallback } from 'react';
import { useWsStore, useDataStore } from '../store';

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();

  // Use selectors — only subscribe to 'connected' for re-render
  const connected = useWsStore((s) => s.connected);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      useWsStore.getState().setConnected(true);
      ws.send(JSON.stringify({
        type: 'subscribe',
        channels: [
          { name: 'ticker', pair: 'BTCUSDT' },
          { name: 'candles_last', pair: 'BTCUSDT', tf: 60 },
          { name: 'orders' },
          { name: 'positions' },
          { name: 'balance' },
          { name: 'alerts' },
          { name: 'pnl' },
        ],
      }));
    };

    ws.onclose = () => {
      useWsStore.getState().setConnected(false);
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = (error) => console.error('WS error:', error);

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        const wsState = useWsStore.getState();
        const dataState = useDataStore.getState();

        switch (message.type) {
          case 'ticker_update':
            wsState.setLastPrice(message.price);
            wsState.setPriceChange24h(message.change_24h);
            wsState.setPriceChange1h(message.change_1h);
            wsState.setVolume24h(message.volume_24h);
            wsState.setHigh24h(message.high_24h);
            wsState.setLow24h(message.low_24h);
            break;
          case 'candle_update':
            dataState.updateCandle({
              t: message.t,
              o: message.o,
              h: message.h,
              l: message.l,
              c: message.c,
              v: message.v,
            });
            break;
          case 'balance_update':
            dataState.setBalance({
              overall: message.overall,
              in_orders: message.in_orders,
              available: message.available,
            });
            break;
          case 'positions_update':
            dataState.setPositions(message.positions || []);
            break;
          case 'alert_event':
            // Prepend new alert to existing list
            dataState.setAlerts([message, ...dataState.alerts.slice(0, 49)]);
            break;
          case 'pnl_update':
            dataState.setPnlOverview(message);
            break;
        }
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendMessage = useCallback((msg: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { sendMessage, connected };
}
