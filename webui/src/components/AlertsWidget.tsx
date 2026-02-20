import { useEffect } from 'react';
import { useDataStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { Bell, Settings } from 'lucide-react';

export default function AlertsWidget() {
  const { alerts, setAlerts } = useDataStore();

  const { data } = useQuery({
    queryKey: ['alerts'],
    queryFn: () => apiService.getAlerts(10),
    refetchInterval: 10000,
  });

  useEffect(() => {
    if (data) setAlerts(data);
  }, [data]);

  const currentAlerts = alerts || data || [];

  return (
    <div className="card m-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Bell className="w-4 h-4 text-textSecondary" />
          <h3 className="text-sm font-medium text-textPrimary">Alerts</h3>
        </div>
        <button className="p-1 hover:bg-panelAlt rounded">
          <Settings className="w-3 h-3 text-textSecondary" />
        </button>
      </div>

      <div className="space-y-2 max-h-48 overflow-y-auto">
        {currentAlerts.map((alert: any) => (
          <div key={alert.id} className="p-2 rounded bg-panelAlt border border-border hover:border-binanceYellow transition-colors cursor-pointer">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium text-textPrimary">{alert.pair}</span>
              <span className="text-xs text-textSecondary">{alert.time_ago}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className={`text-xs ${alert.alert_type === 'Long' ? 'text-bull' : alert.alert_type === 'Short' ? 'text-bear' : 'text-textSecondary'}`}>
                {alert.alert_type}
              </span>
              <span className={`text-xs ${alert.severity === 'critical' ? 'text-bear' : alert.severity === 'warning' ? 'text-binanceYellow' : 'text-textSecondary'}`}>
                {alert.severity}
              </span>
            </div>
          </div>
        ))}
        {currentAlerts.length === 0 && (
          <div className="text-center text-textSecondary text-xs py-4">No alerts</div>
        )}
      </div>
    </div>
  );
}
