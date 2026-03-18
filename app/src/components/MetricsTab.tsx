import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import { Activity, Zap, Clock, AlertCircle } from 'lucide-react'
import { useMetrics } from '../hooks/useMetrics'
import type { MetricsSnapshot } from '../types'

const fmt = (ts: number) => new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
const ms = (s: number) => `${(s * 1000).toFixed(0)}ms`
const tok = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n))

interface StatCardProps {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  accent?: string
}

function StatCard({ icon, label, value, sub, accent = 'text-violet-400' }: StatCardProps) {
  return (
    <div className="bg-surface border border-border rounded-xl p-4">
      <div className={`flex items-center gap-2 text-xs text-slate-500 mb-2`}>
        <span className={accent}>{icon}</span>
        {label}
      </div>
      <p className="text-2xl font-mono font-medium text-slate-100">{value}</p>
      {sub && <p className="text-xs text-slate-600 mt-0.5">{sub}</p>}
    </div>
  )
}

interface ChartCardProps {
  title: string
  data: MetricsSnapshot[]
  lines: { key: keyof MetricsSnapshot; label: string; color: string; format?: (v: number) => string }[]
  yFormat?: (v: number) => string
}

function ChartCard({ title, data, lines, yFormat }: ChartCardProps) {
  return (
    <div className="bg-surface border border-border rounded-xl p-4">
      <h3 className="text-xs font-medium text-slate-400 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 2, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={fmt}
            tick={{ fill: '#475569', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tickFormatter={yFormat}
            tick={{ fill: '#475569', fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={40}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#13131a', border: '1px solid #1e1e2e', borderRadius: 8, fontSize: 11 }}
            labelFormatter={v => new Date(v as number).toLocaleTimeString()}
            formatter={(value: number, name: string) => {
              const line = lines.find(l => l.label === name)
              return [line?.format ? line.format(value) : value.toFixed(3), name]
            }}
          />
          {lines.length > 1 && (
            <Legend wrapperStyle={{ fontSize: 11, color: '#64748b', paddingTop: 8 }} />
          )}
          {lines.map(l => (
            <Line
              key={l.key as string}
              type="monotone"
              dataKey={l.key as string}
              name={l.label}
              stroke={l.color}
              dot={false}
              strokeWidth={1.5}
              activeDot={{ r: 3 }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export function MetricsTab() {
  const { history, error } = useMetrics(3000)

  const latest = history[history.length - 1]

  const errorRate =
    latest && latest.requestsSuccess + latest.requestsError > 0
      ? ((latest.requestsError / (latest.requestsSuccess + latest.requestsError)) * 100).toFixed(1)
      : '0.0'

  return (
    <div className="flex flex-col h-full overflow-y-auto px-4 py-4 gap-4">
      {error && (
        <div className="flex items-center gap-2 text-xs text-amber-400 bg-amber-950/30 border border-amber-500/20 rounded-lg px-3 py-2">
          <AlertCircle size={13} />
          Cannot reach /metrics — is the server running? ({error})
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-4 gap-3">
        <StatCard
          icon={<Activity size={13} />}
          label="In flight"
          value={String(latest?.requestsInFlight ?? 0)}
          sub="active requests"
          accent="text-emerald-400"
        />
        <StatCard
          icon={<Zap size={13} />}
          label="Requests"
          value={String(latest?.requestsSuccess ?? 0)}
          sub={`${latest?.requestsError ?? 0} errors · ${errorRate}% err rate`}
          accent="text-blue-400"
        />
        <StatCard
          icon={<Clock size={13} />}
          label="Avg latency"
          value={latest ? ms(latest.latencyAvg) : '—'}
          sub={`TTFT ${latest ? ms(latest.ttftAvg) : '—'}`}
          accent="text-violet-400"
        />
        <StatCard
          icon={<Zap size={13} />}
          label="Throughput"
          value={latest ? `${Math.round(latest.completionTokensRate)} tok/s` : '—'}
          sub={`${tok(latest?.completionTokensTotal ?? 0)} total generated`}
          accent="text-orange-400"
        />
      </div>

      {/* Charts grid */}
      <div className="grid grid-cols-2 gap-3">
        <ChartCard
          title="Requests in flight"
          data={history}
          lines={[{ key: 'requestsInFlight', label: 'In flight', color: '#22c55e' }]}
          yFormat={v => String(Math.round(v))}
        />
        <ChartCard
          title="Token throughput (tok/s)"
          data={history}
          lines={[
            { key: 'completionTokensRate', label: 'Completion', color: '#7c3aed', format: v => `${Math.round(v)} tok/s` },
            { key: 'promptTokensRate', label: 'Prompt', color: '#3b82f6', format: v => `${Math.round(v)} tok/s` },
          ]}
          yFormat={v => `${Math.round(v)}`}
        />
        <ChartCard
          title="Avg request latency (ms)"
          data={history}
          lines={[{ key: 'latencyAvg', label: 'Avg latency', color: '#7c3aed', format: ms }]}
          yFormat={v => ms(v)}
        />
        <ChartCard
          title="Avg time to first token (ms)"
          data={history}
          lines={[{ key: 'ttftAvg', label: 'Avg TTFT', color: '#f59e0b', format: ms }]}
          yFormat={v => ms(v)}
        />
      </div>

      <p className="text-[10px] text-slate-700 text-center pb-2">
        Polling /metrics every 3s · {history.length} data points
      </p>
    </div>
  )
}
