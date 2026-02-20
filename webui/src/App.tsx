import { Panel, Group, Separator } from 'react-resizable-panels';
import { useWebSocket } from './hooks/useWebSocket';
import TopBar from './components/TopBar';
import TickerStrip from './components/TickerStrip';
import ChartPanel from './components/ChartPanel';
import PositionsPanel from './components/PositionsPanel';
import PnlWidget from './components/PnlWidget';
import BalanceWidget from './components/BalanceWidget';
import AlertsWidget from './components/AlertsWidget';
import TradePanel from './components/TradePanel';
import BottomBar from './components/BottomBar';
import OptionsModal from './components/modals/OptionsModal';
import StrategiesModal from './components/modals/StrategiesModal';
import SignalsModal from './components/modals/SignalsModal';

function App() {
  useWebSocket();

  return (
    <div className="h-screen w-screen bg-background flex flex-col overflow-hidden">
      <TopBar />
      <TickerStrip />
      
      {/* Main Content - Resizable Panels */}
      <Group orientation="horizontal" className="flex-1 min-h-0" autoSave="webui-layout">
        {/* Center Column - Chart + Positions */}
        <Panel defaultSize={55} minSize={30} className="flex flex-col min-w-0" collapsible={false}>
          <Group orientation="vertical" className="h-full" autoSave="webui-left">
            <Panel defaultSize={60} minSize={20}>
              <ChartPanel />
            </Panel>
            <Separator className="h-1 bg-border hover:bg-binanceYellow transition-colors cursor-row-resize" />
            <Panel>
              <PositionsPanel />
            </Panel>
          </Group>
        </Panel>
        
        {/* Resize Handle */}
        <Separator className="w-1 bg-border hover:bg-binanceYellow transition-colors cursor-col-resize" />
        
        {/* Middle Right - PnL, Balance, Alerts */}
        <Panel defaultSize={18} minSize={15} className="flex flex-col min-w-0" collapsible={false}>
          <Group orientation="vertical" className="h-full" autoSave="webui-middle">
            <Panel defaultSize={35} minSize={15}>
              <PnlWidget />
            </Panel>
            <Separator className="h-1 bg-border hover:bg-binanceYellow transition-colors cursor-row-resize" />
            <Panel defaultSize={35} minSize={15}>
              <BalanceWidget />
            </Panel>
            <Separator className="h-1 bg-border hover:bg-binanceYellow transition-colors cursor-row-resize" />
            <Panel>
              <AlertsWidget />
            </Panel>
          </Group>
        </Panel>
        
        {/* Resize Handle */}
        <Separator className="w-1 bg-border hover:bg-binanceYellow transition-colors cursor-col-resize" />
        
        {/* Right Column - Trading Panel */}
        <Panel defaultSize={27} minSize={20} className="p-3 overflow-y-auto" collapsible={false}>
          <TradePanel />
        </Panel>
      </Group>
      
      <BottomBar />
      <OptionsModal />
      <StrategiesModal />
      <SignalsModal />
    </div>
  );
}

export default App;
