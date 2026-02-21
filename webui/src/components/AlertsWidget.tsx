import { useEffect } from 'react';
import { useDataStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { Bell, AlertTriangle, Info, XCircle } from 'lucide-react';

export default function AlertsWidget() {
  const { alerts, setAlerts } = useDataStore();

  // Fetch alerts from risk.alerts table (risk_manager module)
  const { data } = useQuery({
    queryKey: ['alerts'],
    queryFn: () => apiService.getAlerts(20),
    refetchInterval: 10000,
  });

  useEffect(() => {
    if (data) setAlerts(data);
  }, [data]);

  const currentAlerts = alerts.length > 0 ? alerts : (data || []);

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <XCircle className="w-3 h-3 text-bear" />;
      case 'warning': return <AlertTriangle className="w-3 h-3 text-binanceYellow" />;
      default: return <Info className="w-3 h-3 text-textSecondary" />;
    }
  };

  const getSeverityBorder = (severity: string) => {
    switch (severity) {
      case 'critical': return 'border-l-2 border-l-bear';
      case 'warning': return 'border-l-2 border-l-binanceYellow';
      default: return 'border-l-2 border-l-border';
    }
  };

  return (
    <div className="card m-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Bell className="w-4 h-4 text-textSecondary" />
          <h3 className="text-sm font-medium text-textPrimary">Alerts</h3>
          {currentAlerts.length > 0 && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-bear/20 text-bear">
              {currentAlerts.length}
            </span>
          )}
        </div>
      </div>

      <div className="space-y-1.5 max-h-48 overflow-y-auto">
        {currentAlerts.map((alert: any, idx: number) => (
          <div
            key={alert.id || idx}
            className={`p-2 rounded bg-panelAlt ${getSeverityBorder(alert.severity)} hover:border-binanceYellow transition-colors cursor-pointer`}
          >
            <div className="flex items-center justify-between mb-0.5">
              <div className="flex items-center gap-1.5">
                {getSeverityIcon(alert.severity)}
                <span className="text-xs font-medium text-textPrimary">{alert.pair}</span>
              </div>
              <span className="text-xs text-textSecondary">{alert.time_ago}</span>
            </div>
            <p className="text-xs text-textSecondary truncate">{alert.message}</p>
            {alert.source && (
              <span className="text-xs text-textSecondary opacity-50">{alert.source}</span>
            )}
          </div>
        ))}
        {currentAlerts.length === 0 && (
          <div className="text-center text-textSecondary text-xs py-4">No alerts</div>
        )}
      </div>
    </div>
  );
}
