import { Panel, Group, Separator } from 'react-resizable-panels';
import { useWebSocket } from './hooks/useWebSocket';
import TopBar from './components/TopBar';
import TickerStrip from './components/TickerStrip';
import ChartPanel from './components/ChartPanel';
import PositionsPanel from './components/PositionsPanel';
import PnlWidget from './components/PnlWidget';
import BalanceWidget from './components/BalanceWidget';
import ConnectionsWidget from './components/ConnectionsWidget';
import AlertsWidget from './components/AlertsWidget';
import TerminalWidget from './components/TerminalWidget';
import TradePanel from './components/TradePanel';
import BottomBar from './components/BottomBar';
import TradingOptionsModal from './components/modals/TradingOptionsModal';
import OrderOptionsModal from './components/modals/OrderOptionsModal';
import SignalsModal from './components/modals/SignalsModal';
import StatisticsPanel from './components/StatisticsPanel';
import AccountModal from './components/AccountModal';
import { Component, useEffect } from 'react';
import type { ReactNode, ErrorInfo } from 'react';

// ─── ErrorBoundary — prevents blank screen on unhandled React errors ───
interface ErrorBoundaryState { hasError: boolean; error: Error | null }
class ErrorBoundary extends Component<{ children: ReactNode }, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false, error: null };
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary] Caught rendering error:', error, info.componentStack);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen w-screen bg-background flex items-center justify-center flex-col gap-4">
          <div className="text-xl text-red-400 font-bold">⚠️ UI Error</div>
          <div className="text-sm text-textSecondary max-w-xl text-center">
            {this.state.error?.message || 'Unknown rendering error'}
          </div>
          <button
            className="px-4 py-2 bg-binanceYellow text-black rounded font-medium"
            onClick={() => { this.setState({ hasError: false, error: null }); }}
          >
            Reload UI
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  useWebSocket();

  // Clear saved layout on first load to ensure proper default sizes
  useEffect(() => {
    const hasVisited = localStorage.getItem('webui-visited-v2');
    if (!hasVisited) {
      localStorage.removeItem('webui-layout');
      localStorage.removeItem('webui-left');
      localStorage.removeItem('webui-middle');
      localStorage.setItem('webui-visited-v2', 'true');
    }
  }, []);

  return (
    <div className="h-screen w-screen bg-background flex flex-col overflow-hidden">
      <TopBar />
      <TickerStrip />

      {/* Main Content - Resizable Panels */}
      <Group
        orientation="horizontal"
        className="flex-1 min-h-0"
        autoSave="webui-layout"
      >
        {/* Center Column - Chart + Positions */}
        <Panel defaultSize={55} minSize={25} className="flex flex-col min-w-0">
          <Group
            orientation="vertical"
            className="h-full"
            autoSave="webui-left"
          >
            <Panel defaultSize={65} minSize={20}>
              <ChartPanel />
            </Panel>
            <Separator className="h-1 bg-border hover:bg-binanceYellow transition-colors cursor-row-resize" />
            <Panel defaultSize={35} minSize={15}>
              <PositionsPanel />
            </Panel>
          </Group>
        </Panel>

        <Separator className="w-1 bg-border hover:bg-binanceYellow transition-colors cursor-col-resize" />

        {/* Middle Right - PnL, Balance, Connections, Alerts */}
        <Panel defaultSize={18} minSize={15} className="flex flex-col min-w-0">
          <Group
            orientation="vertical"
            className="h-full"
            autoSave="webui-middle"
          >
            <Panel defaultSize={22} minSize={8}>
              <PnlWidget />
            </Panel>
            <Separator className="h-1 bg-border hover:bg-binanceYellow transition-colors cursor-row-resize" />
            <Panel defaultSize={26} minSize={8}>
              <BalanceWidget />
            </Panel>
            <Separator className="h-1 bg-border hover:bg-binanceYellow transition-colors cursor-row-resize" />
            <Panel defaultSize={22} minSize={8}>
              <ConnectionsWidget />
            </Panel>
            <Separator className="h-1 bg-border hover:bg-binanceYellow transition-colors cursor-row-resize" />
            <Panel defaultSize={22} minSize={8}>
              <AlertsWidget />
            </Panel>
            <Separator className="h-1 bg-border hover:bg-binanceYellow transition-colors cursor-row-resize" />
            <Panel defaultSize={18} minSize={8}>
              <TerminalWidget />
            </Panel>
          </Group>
        </Panel>

        <Separator className="w-1 bg-border hover:bg-binanceYellow transition-colors cursor-col-resize" />

        {/* Right Column - Trading Panel */}
        <Panel defaultSize={27} minSize={20} className="p-3 overflow-y-auto">
          <TradePanel />
        </Panel>
      </Group>

      <BottomBar />
      <TradingOptionsModal />
      <OrderOptionsModal />
      <SignalsModal />
      <StatisticsPanel />
      <AccountModal />
    </div>
  );
}

function AppWithErrorBoundary() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  );
}

export default AppWithErrorBoundary;
