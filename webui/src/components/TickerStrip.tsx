export default function TickerStrip() {
  const TICKERS = [
    { pair: 'ETHUSDT', change: 3.42 },
    { pair: 'BNBUSDT', change: 1.25 },
    { pair: 'SOLUSDT', change: -2.15 },
    { pair: 'XRPUSDT', change: 0.87 },
    { pair: 'ADAUSDT', change: -1.23 },
    { pair: 'DOGEUSDT', change: 5.67 },
    { pair: 'AVAXUSDT', change: 2.34 },
    { pair: 'DOTUSDT', change: -0.45 },
  ];

  return (
    <div className="h-8 bg-panelAlt border-b border-border flex items-center px-4 gap-4 overflow-x-auto shrink-0">
      {TICKERS.map((ticker) => (
        <div key={ticker.pair} className="flex items-center gap-2 text-xs whitespace-nowrap">
          <span className="text-textSecondary">{ticker.pair}</span>
          <span className={ticker.change >= 0 ? 'text-bull' : 'text-bear'}>
            {ticker.change >= 0 ? '+' : ''}{ticker.change.toFixed(2)}%
          </span>
        </div>
      ))}
    </div>
  );
}
