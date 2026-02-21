import { useTradingStore, useUiStore, useDataStore, useWsStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { ChevronDown, Bell, Settings, Grid3x3, User, SquareTerminal } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';

export default function TopBar() {
  const { currentPair, setCurrentPair } = useTradingStore();
  const { setTradingOptionsModalOpen, setOrderOptionsModalOpen, setSignalsModalOpen } = useUiStore();
  const connections = useDataStore((s) => s.connections);
  const { connected, lastPrice, priceChange24h, priceChange1h, volume24h, high24h, low24h } = useWsStore();
  const [pairSearchOpen, setPairSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Load ALL pairs from DB (no limit)
  const { data: pairs = [] } = useQuery({
    queryKey: ['pairs', searchQuery],
    queryFn: () => apiService.getPairs(searchQuery, 500),
    enabled: pairSearchOpen,
  });

  const { data: summary } = useQuery({
    queryKey: ['market-summary', currentPair],
    queryFn: () => apiService.getMarketSummary(currentPair, 60),
    refetchInterval: 5000,
  });

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setPairSearchOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const price = summary?.price || lastPrice || 0;
  const change24h = summary?.change_24h || priceChange24h;
  const change1h = summary?.change_1h || priceChange1h;

  // Connection indicator color logic:
  // All connected → green
  // WS or Account disconnected → red
  // Any other single disconnected → yellow
  const getConnectionColor = () => {
    if (!connections) return 'bg-textSecondary';
    const { database, redpanda, rest_api, websocket, account } = connections;
    const allConnected = database && redpanda && rest_api && websocket && account;
    if (allConnected) return 'bg-bull';
    if (!websocket || !account) return 'bg-bear';
    return 'bg-binanceYellow';
  };

  const getConnectionLabel = () => {
    if (!connections) return 'Checking...';
    const { database, redpanda, rest_api, websocket, account } = connections;
    const allConnected = database && redpanda && rest_api && websocket && account;
    if (allConnected) return 'Connections: OK';
    if (!websocket || !account) return 'Connections: Critical';
    return 'Connections: Partial';
  };

  return (
    <header className="h-14 bg-panel border-b border-border flex items-center px-4 justify-between shrink-0">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-binanceYellow rounded-lg flex items-center justify-center">
            <SquareTerminal className="w-5 h-5 text-black" />
          </div>
          <div>
            <div className="relative" ref={dropdownRef}>
              <button
                className="text-xl font-bold text-textPrimary hover:text-binanceYellow transition-colors flex items-center gap-1"
                onClick={() => setPairSearchOpen(!pairSearchOpen)}
              >
                {currentPair}
                <ChevronDown className="w-4 h-4" />
              </button>

              {pairSearchOpen && (
                <div className="absolute top-full left-0 mt-1 w-72 bg-panel border border-border rounded-lg shadow-xl z-50">
                  <input
                    type="text"
                    placeholder="Search pair..."
                    className="w-full bg-panelAlt border-b border-border px-3 py-2 text-sm focus:outline-none focus:border-binanceYellow"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    autoFocus
                  />
                  <div className="max-h-80 overflow-y-auto">
                    {pairs.map((pair: any) => (
                      <button
                        key={pair.symbol_id}
                        className="w-full px-3 py-2 text-left hover:bg-panelAlt text-sm flex justify-between"
                        onClick={() => {
                          setCurrentPair(pair.symbol);
                          setPairSearchOpen(false);
                          setSearchQuery('');
                        }}
                      >
                        <span>{pair.symbol}</span>
                      </button>
                    ))}
                    {pairs.length === 0 && (
                      <div className="px-3 py-4 text-center text-textSecondary text-xs">No pairs found</div>
                    )}
                  </div>
                </div>
              )}
            </div>
            <div className={`text-sm ${change24h >= 0 ? 'text-bull' : 'text-bear'}`}>
              ${price?.toFixed(2)}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs">
          <div>
            <div className="text-textSecondary">Volume (24h)</div>
            <div className="text-textPrimary">${(summary?.volume_24h || volume24h || 0).toLocaleString()}</div>
          </div>
          <div>
            <div className="text-textSecondary">Change (24h)</div>
            <div className={change24h >= 0 ? 'text-bull' : 'text-bear'}>{change24h?.toFixed(2)}%</div>
          </div>
          <div>
            <div className="text-textSecondary">Change (1h)</div>
            <div className={change1h >= 0 ? 'text-bull' : 'text-bear'}>{change1h?.toFixed(2)}%</div>
          </div>
          <div>
            <div className="text-textSecondary">High (24h)</div>
            <div className="text-textPrimary">${(summary?.high_24h || high24h || 0).toFixed(2)}</div>
          </div>
          <div>
            <div className="text-textSecondary">Low (24h)</div>
            <div className="text-textPrimary">${(summary?.low_24h || low24h || 0).toFixed(2)}</div>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-1">
        <button className="px-3 py-2 hover:bg-panelAlt rounded-lg text-sm flex items-center gap-1" onClick={() => setSignalsModalOpen(true)}>
          Signals
        </button>
        <button className="px-3 py-2 hover:bg-panelAlt rounded-lg text-sm flex items-center gap-1" onClick={() => setTradingOptionsModalOpen(true)}>
          Trading Options
          <ChevronDown className="w-3 h-3" />
        </button>
        <button className="px-3 py-2 hover:bg-panelAlt rounded-lg text-sm flex items-center gap-1" onClick={() => setOrderOptionsModalOpen(true)}>
          Order Options
          <ChevronDown className="w-3 h-3" />
        </button>
      </div>

      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1 px-2 py-1 rounded-full bg-panelAlt">
          <div className={`w-2 h-2 rounded-full ${getConnectionColor()}`} />
          <span className="text-xs text-textSecondary">{getConnectionLabel()}</span>
        </div>
        <button className="p-2 hover:bg-panelAlt rounded-lg"><Bell className="w-4 h-4 text-textSecondary" /></button>
        <button className="p-2 hover:bg-panelAlt rounded-lg"><Grid3x3 className="w-4 h-4 text-textSecondary" /></button>
        <button className="p-2 hover:bg-panelAlt rounded-lg"><Settings className="w-4 h-4 text-textSecondary" /></button>
        <button className="p-2 hover:bg-panelAlt rounded-lg"><User className="w-4 h-4 text-textSecondary" /></button>
      </div>
    </header>
  );
}
