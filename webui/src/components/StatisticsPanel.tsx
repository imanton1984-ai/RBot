import { useQuery } from '@tanstack/react-query';
import { apiService } from '../api';
import type { FullStatistics, TimeSlotStats } from '../types';
import { useState } from 'react';
import { X, TrendingUp, TrendingDown, Target, Clock, Award, Calendar, BarChart3, PieChart } from 'lucide-react';
import { useUiStore } from '../store';
import {
  LineChart, Line, BarChart, Bar, PieChart as RechartsPie, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area
} from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FF6B6B', '#4ECDC4'];

export default function StatisticsPanel() {
  const { statisticsModalOpen, setStatisticsModalOpen } = useUiStore();
  const [timeRange, setTimeRange] = useState('30d');
  const [activeTab, setActiveTab] = useState<'overview' | 'time' | 'distribution'>('overview');

  const { data: stats, isLoading } = useQuery({
    queryKey: ['statistics', timeRange],
    queryFn: () => apiService.getStatistics(timeRange, 'daily'),
    enabled: statisticsModalOpen,
    refetchInterval: 60000,
  });

  if (!statisticsModalOpen) return null;

  const summary = stats?.summary;
  const timeline = stats?.timeline;
  const distribution = stats?.distribution;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-8">
      <div className="bg-panel rounded-2xl w-full max-w-7xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center gap-3">
            <BarChart3 className="w-8 h-8 text-binanceYellow" />
            <h2 className="text-2xl font-bold text-textPrimary">Trading Statistics</h2>
          </div>
          <button
            onClick={() => setStatisticsModalOpen(false)}
            className="p-2 hover:bg-panelAlt rounded-lg transition-colors"
          >
            <X className="w-6 h-6 text-textSecondary" />
          </button>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between p-4 border-b border-border gap-4">
          <div className="flex items-center gap-2">
            {['24h', '7d', '30d', '90d', 'all'].map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  timeRange === range
                    ? 'bg-binanceYellow text-black'
                    : 'bg-panelAlt text-textSecondary hover:text-textPrimary'
                }`}
              >
                {range === 'all' ? 'All Time' : range.toUpperCase()}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1 bg-panelAlt rounded-lg p-1">
            <button
              onClick={() => setActiveTab('overview')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'overview' ? 'bg-binanceYellow text-black' : 'text-textSecondary'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('time')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'time' ? 'bg-binanceYellow text-black' : 'text-textSecondary'
              }`}
            >
              Time Analysis
            </button>
            <button
              onClick={() => setActiveTab('distribution')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'distribution' ? 'bg-binanceYellow text-black' : 'text-textSecondary'
              }`}
            >
              Distribution
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-binanceYellow"></div>
            </div>
          ) : stats ? (
            <>
              {activeTab === 'overview' && (
                <OverviewTab summary={summary} timeline={timeline} bestWorst={stats.best_worst} />
              )}
              {activeTab === 'time' && (
                <TimeTab hourly={stats.hourly} daily={stats.daily} weekly={stats.weekly} monthly={stats.monthly} />
              )}
              {activeTab === 'distribution' && (
                <DistributionTab distribution={distribution} />
              )}
            </>
          ) : (
            <div className="text-center text-textSecondary py-12">
              No statistics available for the selected period
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── OVERVIEW TAB ────────────────────────────────────────────
interface OverviewTabProps {
  summary?: FullStatistics['summary'];
  timeline?: FullStatistics['timeline'];
  bestWorst?: FullStatistics['best_worst'];
}

function OverviewTab({ summary, timeline, bestWorst }: OverviewTabProps) {
  if (!summary) return null;

  const statCards = [
    { label: 'Total Trades', value: summary.total_trades.toString(), icon: Clock },
    { label: 'Win Rate', value: `${summary.win_rate.toFixed(1)}%`, icon: Target },
    { label: 'Total PnL', value: `${summary.total_pnl >= 0 ? '+' : ''}${summary.total_pnl.toFixed(2)} USDT`, icon: summary.total_pnl >= 0 ? TrendingUp : TrendingDown, color: summary.total_pnl >= 0 ? 'text-bull' : 'text-bear' },
    { label: 'Profit Factor', value: summary.profit_factor.toFixed(2), icon: BarChart3 },
    { label: 'Avg Win', value: `${summary.avg_win.toFixed(2)} USDT`, icon: Award, color: 'text-bull' },
    { label: 'Avg Loss', value: `${summary.avg_loss.toFixed(2)} USDT`, icon: Award, color: 'text-bear' },
    { label: 'Best Trade', value: `${summary.best_trade >= 0 ? '+' : ''}${summary.best_trade.toFixed(2)} USDT`, icon: TrendingUp, color: 'text-bull' },
    { label: 'Worst Trade', value: `${summary.worst_trade >= 0 ? '+' : ''}${summary.worst_trade.toFixed(2)} USDT`, icon: TrendingDown, color: 'text-bear' },
    { label: 'Avg Duration', value: `${summary.avg_trade_duration_hours.toFixed(1)}h`, icon: Clock },
    { label: 'Max Win Streak', value: summary.max_consecutive_wins.toString(), icon: Award, color: 'text-bull' },
    { label: 'Max Loss Streak', value: summary.max_consecutive_losses.toString(), icon: TrendingDown, color: 'text-bear' },
  ];

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        {statCards.map((stat) => {
          const Icon = stat.icon;
          return (
            <div key={stat.label} className="bg-panelAlt rounded-xl p-4">
              <div className="flex items-center gap-2 mb-2">
                <Icon className={`w-4 h-4 ${stat.color || 'text-textSecondary'}`} />
                <span className="text-xs text-textSecondary">{stat.label}</span>
              </div>
              <div className={`text-lg font-bold ${stat.color || 'text-textPrimary'}`}>
                {stat.value}
              </div>
            </div>
          );
        })}
      </div>

      {/* PnL Timeline Chart */}
      {timeline && timeline.points.length > 0 && (
        <div className="bg-panelAlt rounded-xl p-6">
          <h3 className="text-lg font-semibold text-textPrimary mb-4">Cumulative PnL</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={timeline.points}>
              <defs>
                <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={timeline.points[timeline.points.length - 1]?.cumulative_pnl >= 0 ? '#00C49F' : '#FF6B6B'} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={timeline.points[timeline.points.length - 1]?.cumulative_pnl >= 0 ? '#00C49F' : '#FF6B6B'} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="t" 
                tickFormatter={(t) => new Date(t).toLocaleDateString()}
                stroke="#6B7280"
                fontSize={12}
              />
              <YAxis stroke="#6B7280" fontSize={12} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                labelFormatter={(t) => new Date(t).toLocaleDateString()}
                formatter={(value: number) => [`${value >= 0 ? '+' : ''}${value.toFixed(2)} USDT`, 'PnL']}
              />
              <Area 
                type="monotone" 
                dataKey="cumulative_pnl" 
                stroke={timeline.points[timeline.points.length - 1]?.cumulative_pnl >= 0 ? '#00C49F' : '#FF6B6B'}
                fill="url(#pnlGradient)"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Best & Worst Trades */}
      {bestWorst && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-panelAlt rounded-xl p-6">
            <h3 className="text-lg font-semibold text-bull mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" /> Best Trades
            </h3>
            <div className="space-y-2">
              {bestWorst.best.map((trade, i) => (
                <div key={i} className="flex justify-between items-center p-3 bg-panel rounded-lg">
                  <div>
                    <div className="text-sm font-medium text-textPrimary">{trade.pair}</div>
                    <div className="text-xs text-textSecondary">{trade.side} • {new Date(trade.close_time).toLocaleDateString()}</div>
                  </div>
                  <div className="text-bull font-semibold">+{trade.pnl_usdt.toFixed(2)} USDT</div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-panelAlt rounded-xl p-6">
            <h3 className="text-lg font-semibold text-bear mb-4 flex items-center gap-2">
              <TrendingDown className="w-5 h-5" /> Worst Trades
            </h3>
            <div className="space-y-2">
              {bestWorst.worst.map((trade, i) => (
                <div key={i} className="flex justify-between items-center p-3 bg-panel rounded-lg">
                  <div>
                    <div className="text-sm font-medium text-textPrimary">{trade.pair}</div>
                    <div className="text-xs text-textSecondary">{trade.side} • {new Date(trade.close_time).toLocaleDateString()}</div>
                  </div>
                  <div className="text-bear font-semibold">{trade.pnl_usdt.toFixed(2)} USDT</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── TIME TAB ────────────────────────────────────────────
interface TimeTabProps {
  hourly?: FullStatistics['hourly'];
  daily?: FullStatistics['daily'];
  weekly?: FullStatistics['weekly'];
  monthly?: FullStatistics['monthly'];
}

function TimeTab({ hourly, daily, weekly, monthly }: TimeTabProps) {
  const [timeView, setTimeView] = useState<'hourly' | 'daily' | 'weekly' | 'monthly'>('daily');

  const getData = () => {
    switch (timeView) {
      case 'hourly': return hourly?.data || [];
      case 'daily': return daily?.data || [];
      case 'weekly': return weekly?.data || [];
      case 'monthly': return monthly?.map(m => ({
        label: m.month,
        trades: m.trades,
        wins: m.wins,
        losses: m.trades - m.wins,
        pnl: m.pnl,
        win_rate: m.win_rate,
        avg_pnl: m.trades > 0 ? m.pnl / m.trades : 0,
      })) || [];
    }
  };

  const data = getData();

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        {(['hourly', 'daily', 'weekly', 'monthly'] as const).map((view) => (
          <button
            key={view}
            onClick={() => setTimeView(view)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              timeView === view
                ? 'bg-binanceYellow text-black'
                : 'bg-panelAlt text-textSecondary hover:text-textPrimary'
            }`}
          >
            {view.charAt(0).toUpperCase() + view.slice(1)}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Win Rate by Period */}
        <div className="bg-panelAlt rounded-xl p-6">
          <h3 className="text-lg font-semibold text-textPrimary mb-4">Win Rate by {timeView}</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="label" stroke="#6B7280" fontSize={12} angle={-45} textAnchor="end" height={60} />
              <YAxis yAxisId="left" stroke="#6B7280" fontSize={12} />
              <YAxis yAxisId="right" orientation="right" stroke="#6B7280" fontSize={12} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
              />
              <Legend />
              <Bar yAxisId="left" dataKey="trades" fill="#3B82F6" name="Trades" />
              <Bar yAxisId="right" dataKey="win_rate" fill="#10B981" name="Win Rate %" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* PnL by Period */}
        <div className="bg-panelAlt rounded-xl p-6">
          <h3 className="text-lg font-semibold text-textPrimary mb-4">PnL by {timeView}</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="label" stroke="#6B7280" fontSize={12} angle={-45} textAnchor="end" height={60} />
              <YAxis stroke="#6B7280" fontSize={12} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                formatter={(value: number) => [`${value >= 0 ? '+' : ''}${value.toFixed(2)} USDT`, 'PnL']}
              />
              <Bar dataKey="pnl" name="PnL">
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.pnl >= 0 ? '#10B981' : '#EF4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Table */}
      <div className="bg-panelAlt rounded-xl p-6 overflow-x-auto">
        <h3 className="text-lg font-semibold text-textPrimary mb-4">Detailed Statistics</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left py-3 px-4 text-textSecondary font-medium">Period</th>
              <th className="text-right py-3 px-4 text-textSecondary font-medium">Trades</th>
              <th className="text-right py-3 px-4 text-textSecondary font-medium">Wins</th>
              <th className="text-right py-3 px-4 text-textSecondary font-medium">Losses</th>
              <th className="text-right py-3 px-4 text-textSecondary font-medium">Win Rate</th>
              <th className="text-right py-3 px-4 text-textSecondary font-medium">PnL</th>
              <th className="text-right py-3 px-4 text-textSecondary font-medium">Avg PnL</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i} className="border-b border-border/50 hover:bg-panel">
                <td className="py-3 px-4 text-textPrimary font-medium">{row.label}</td>
                <td className="py-3 px-4 text-right text-textPrimary">{row.trades}</td>
                <td className="py-3 px-4 text-right text-bull">{row.wins}</td>
                <td className="py-3 px-4 text-right text-bear">{row.losses}</td>
                <td className="py-3 px-4 text-right">
                  <span className={row.win_rate >= 50 ? 'text-bull' : 'text-bear'}>
                    {row.win_rate.toFixed(1)}%
                  </span>
                </td>
                <td className={`py-3 px-4 text-right font-medium ${row.pnl >= 0 ? 'text-bull' : 'text-bear'}`}>
                  {row.pnl >= 0 ? '+' : ''}{row.pnl.toFixed(2)}
                </td>
                <td className={`py-3 px-4 text-right ${row.avg_pnl >= 0 ? 'text-bull' : 'text-bear'}`}>
                  {row.avg_pnl >= 0 ? '+' : ''}{row.avg_pnl.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── DISTRIBUTION TAB ────────────────────────────────────────────
interface DistributionTabProps {
  distribution?: FullStatistics['distribution'];
}

function DistributionTab({ distribution }: DistributionTabProps) {
  if (!distribution) return null;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* By Pair */}
        <div className="bg-panelAlt rounded-xl p-6">
          <h3 className="text-lg font-semibold text-textPrimary mb-4 flex items-center gap-2">
            <Target className="w-5 h-5" /> By Pair
          </h3>
          <div className="space-y-3">
            {distribution.by_pair.map((stat, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-panel rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                  <span className="text-textPrimary font-medium">{stat.pair}</span>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-textSecondary">{stat.trades} trades</span>
                  <span className={stat.win_rate >= 50 ? 'text-bull' : 'text-bear'}>
                    {stat.win_rate.toFixed(1)}%
                  </span>
                  <span className={`font-medium ${stat.pnl >= 0 ? 'text-bull' : 'text-bear'}`}>
                    {stat.pnl >= 0 ? '+' : ''}{stat.pnl.toFixed(2)} USDT
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* By Timeframe */}
        <div className="bg-panelAlt rounded-xl p-6">
          <h3 className="text-lg font-semibold text-textPrimary mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5" /> By Timeframe
          </h3>
          <div className="space-y-3">
            {distribution.by_timeframe.map((stat, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-panel rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                  <span className="text-textPrimary font-medium">{stat.tf}m</span>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-textSecondary">{stat.trades} trades</span>
                  <span className={stat.win_rate >= 50 ? 'text-bull' : 'text-bear'}>
                    {stat.win_rate.toFixed(1)}%
                  </span>
                  <span className={`font-medium ${stat.pnl >= 0 ? 'text-bull' : 'text-bear'}`}>
                    {stat.pnl >= 0 ? '+' : ''}{stat.pnl.toFixed(2)} USDT
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* By Close Type */}
        <div className="bg-panelAlt rounded-xl p-6">
          <h3 className="text-lg font-semibold text-textPrimary mb-4 flex items-center gap-2">
            <Calendar className="w-5 h-5" /> By Close Type
          </h3>
          <div className="space-y-3">
            {distribution.by_close_type.map((stat, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-panel rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                  <span className="text-textPrimary font-medium capitalize">{stat.close_type.replace('_', ' ')}</span>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-textSecondary">{stat.count} trades</span>
                  <span className={stat.win_rate >= 50 ? 'text-bull' : 'text-bear'}>
                    {stat.win_rate.toFixed(1)}%
                  </span>
                  <span className={`font-medium ${stat.pnl >= 0 ? 'text-bull' : 'text-bear'}`}>
                    {stat.pnl >= 0 ? '+' : ''}{stat.pnl.toFixed(2)} USDT
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* By Side */}
        <div className="bg-panelAlt rounded-xl p-6">
          <h3 className="text-lg font-semibold text-textPrimary mb-4 flex items-center gap-2">
            <PieChart className="w-5 h-5" /> By Side (LONG/SHORT)
          </h3>
          <div className="space-y-3">
            {distribution.by_side.map((stat, i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-panel rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                  <span className="text-textPrimary font-medium">{stat.side}</span>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <span className="text-textSecondary">{stat.trades} trades</span>
                  <span className={stat.win_rate >= 50 ? 'text-bull' : 'text-bear'}>
                    {stat.win_rate.toFixed(1)}%
                  </span>
                  <span className={`font-medium ${stat.pnl >= 0 ? 'text-bull' : 'text-bear'}`}>
                    {stat.pnl >= 0 ? '+' : ''}{stat.pnl.toFixed(2)} USDT
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
