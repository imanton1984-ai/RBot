import { useEffect, useRef } from 'react';
import { useDataStore } from '../store';
import { Terminal as TerminalIcon, Trash2 } from 'lucide-react';

export interface TerminalLog {
    id: number;
    time: string;
    level: 'error' | 'warn' | 'info';
    message: string;
}

const levelColors: Record<string, string> = {
    error: 'text-bear',
    warn: 'text-binanceYellow',
    info: 'text-textSecondary',
};

const levelPrefix: Record<string, string> = {
    error: '✗ ERROR',
    warn: '⚠ WARN',
    info: 'ℹ INFO',
};

export default function TerminalWidget() {
    const logs = useDataStore((s) => s.terminalLogs);
    const clearLogs = useDataStore((s) => s.clearTerminalLogs);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom on new logs
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="card m-3 mt-0 flex flex-col" style={{ minHeight: '120px' }}>
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <TerminalIcon className="w-4 h-4 text-textSecondary" />
                    <h3 className="text-sm font-medium text-textPrimary">Terminal</h3>
                    {logs.length > 0 && (
                        <span className="text-xs px-1.5 py-0.5 rounded bg-panelAlt text-textSecondary">
                            {logs.length}
                        </span>
                    )}
                </div>
                {logs.length > 0 && (
                    <button
                        className="p-1 hover:bg-panelAlt rounded text-textSecondary hover:text-bear transition-colors"
                        onClick={clearLogs}
                        title="Clear logs"
                    >
                        <Trash2 className="w-3 h-3" />
                    </button>
                )}
            </div>

            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto font-mono text-xs space-y-0.5 bg-background rounded p-2 max-h-40"
            >
                {logs.length > 0 ? (
                    logs.map((log) => (
                        <div key={log.id} className="flex gap-2">
                            <span className="text-textSecondary shrink-0">{log.time}</span>
                            <span className={`shrink-0 ${levelColors[log.level]}`}>
                                {levelPrefix[log.level]}
                            </span>
                            <span className={levelColors[log.level]}>{log.message}</span>
                        </div>
                    ))
                ) : (
                    <div className="text-textSecondary/50 text-center py-4">No messages</div>
                )}
            </div>
        </div>
    );
}
