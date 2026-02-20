import { useEffect, useRef } from 'react';
import { useWsStore, useDataStore } from '../store';

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<any>();
  
  const { connected, lastPrice, priceChange24h, priceChange1h, volume24h, high24h, low24h,
    setConnected, setLastPrice, setPriceChange24h, setPriceChange1h, setVolume24h, setHigh24h, setLow24h } = useWsStore();
  const { updateCandle, updatePosition, setBalance, setAlerts } = useDataStore();

  const connect = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      setConnected(true);
      wsRef.current?.send(JSON.stringify({
        type: 'subscribe',
        channels: [{ name: 'ticker', pair: 'BTCUSDT' }, { name: 'candles_last', pair: 'BTCUSDT', tf: 60 }],
      }));
    };

    wsRef.current.onclose = () => {
      setConnected(false);
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    wsRef.current.onerror = (error) => console.error('WS error:', error);

    wsRef.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        switch (message.type) {
          case 'ticker_update':
            setLastPrice(message.price);
            setPriceChange24h(message.change_24h);
            setPriceChange1h(message.change_1h);
            setVolume24h(message.volume_24h);
            setHigh24h(message.high_24h);
            setLow24h(message.low_24h);
            break;
          case 'candle_update':
            updateCandle({ t: message.t, o: message.o, h: message.h, l: message.l, c: message.c, v: message.v });
            break;
          case 'balance_update':
            setBalance({ overall: message.overall, in_orders: message.in_orders, available: message.available });
            break;
        }
      } catch (e) { console.error('Parse error:', e); }
    };
  };

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close();
    };
  }, []);

  return { sendMessage: (msg: any) => wsRef.current?.readyState === WebSocket.OPEN && wsRef.current.send(JSON.stringify(msg)), connected };
}
