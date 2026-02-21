import { useEffect } from 'react';
import { useDataStore } from '../store';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import { Plug, Database, Radio, Globe, Wifi, UserCheck } from 'lucide-react';

interface ConnectionItemProps {
    label: string;
    connected: boolean;
    icon: React.ReactNode;
}

function ConnectionItem({ label, connected, icon }: ConnectionItemProps) {
    return (
        <div className="flex items-center justify-between py-1.5">
            <div className="flex items-center gap-2">
                <span className="text-textSecondary">{icon}</span>
                <span className="text-xs text-textSecondary">{label}</span>
            </div>
            <div className={`w-2.5 h-2.5 rounded-full ${connected ? 'bg-bull' : 'bg-bear'} shadow-sm ${connected ? 'shadow-bull/30' : 'shadow-bear/30'}`} />
        </div>
    );
}

export default function ConnectionsWidget() {
    const { connections, setConnections } = useDataStore();

    const { data } = useQuery({
        queryKey: ['connections'],
        queryFn: () => apiService.getConnections(),
        refetchInterval: 15000,
        staleTime: 10000,
    });

    useEffect(() => {
        if (data) setConnections(data);
    }, [data]);

    const conn = connections || data || {
        database: false,
        redpanda: false,
        rest_api: false,
        websocket: false,
        account: false,
    };

    return (
        <div className="card m-3 mb-0">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Plug className="w-4 h-4 text-textSecondary" />
                    <h3 className="text-sm font-medium text-textPrimary">Connections</h3>
                </div>
            </div>

            <div className="space-y-0.5">
                <ConnectionItem
                    label="Database"
                    connected={conn.database}
                    icon={<Database className="w-3 h-3" />}
                />
                <ConnectionItem
                    label="Redpanda (Kafka)"
                    connected={conn.redpanda}
                    icon={<Radio className="w-3 h-3" />}
                />
                <ConnectionItem
                    label="REST API"
                    connected={conn.rest_api}
                    icon={<Globe className="w-3 h-3" />}
                />
                <ConnectionItem
                    label="WebSocket"
                    connected={conn.websocket}
                    icon={<Wifi className="w-3 h-3" />}
                />
                <ConnectionItem
                    label="Account"
                    connected={conn.account}
                    icon={<UserCheck className="w-3 h-3" />}
                />
            </div>
        </div>
    );
}
