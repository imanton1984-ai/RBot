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
import { useEffect } from 'react';

function App() {
  useWebSocket();

  // Clear saved layout on first load to ensure proper default sizes
  useEffect(() => {
    const hasVisited = localStorage.getItem('webui-visited');
    if (!hasVisited) {
      // First visit - clear all saved layouts
      localStorage.removeItem('webui-layout');
      localStorage.removeItem('webui-left');
      localStorage.removeItem('webui-middle');
      localStorage.setItem('webui-visited', 'true');
    }
  }, []);

  return (
    <div className="h-screen w-screen bg-background flex flex-col overflow-hidden">
      <TopBar />
      <TickerStrip />
      
      {/* Main Content - Resizable Panels with default sizes */}
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
        
        {/* Resize Handle */}
        <Separator className="w-1 bg-border hover:bg-binanceYellow transition-colors cursor-col-resize" />
        
        {/* Middle Right - PnL, Balance, Alerts */}
        <Panel defaultSize={18} minSize={15} className="flex flex-col min-w-0">
          <Group 
            orientation="vertical" 
            className="h-full" 
            autoSave="webui-middle"
          >
            <Panel defaultSize={33} minSize={10}>
              <PnlWidget />
            </Panel>
            <Separator className="h-1 bg-border hover:bg-binanceYellow transition-colors cursor-row-resize" />
            <Panel defaultSize={33} minSize={10}>
              <BalanceWidget />
            </Panel>
            <Separator className="h-1 bg-border hover:bg-binanceYellow transition-colors cursor-row-resize" />
            <Panel defaultSize={34} minSize={10}>
              <AlertsWidget />
            </Panel>
          </Group>
        </Panel>
        
        {/* Resize Handle */}
        <Separator className="w-1 bg-border hover:bg-binanceYellow transition-colors cursor-col-resize" />
        
        {/* Right Column - Trading Panel */}
        <Panel defaultSize={27} minSize={20} className="p-3 overflow-y-auto">
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
