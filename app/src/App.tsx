import { useState } from 'react'
import { MessageSquare, BarChart2 } from 'lucide-react'
import type { Tab } from './types'
import { ChatTab } from './components/ChatTab'
import { MetricsTab } from './components/MetricsTab'

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')

  return (
    <div className="flex flex-col h-screen bg-[#0a0a0f]">
      {/* Header */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-border flex-shrink-0">
        <div className="flex items-center gap-2.5">
          <span className="text-violet-400 font-mono font-semibold text-sm tracking-tight">
            ◈ GPT-OSS
          </span>
          <span className="text-[10px] text-slate-600 font-mono bg-surface px-2 py-0.5 rounded-full border border-border">
            20B · vLLM
          </span>
        </div>

        {/* Tab bar */}
        <nav className="flex gap-1 bg-surface rounded-lg p-1 border border-border">
          {(
            [
              { id: 'chat' as Tab, label: 'Chat', icon: <MessageSquare size={13} /> },
              { id: 'metrics' as Tab, label: 'Metrics', icon: <BarChart2 size={13} /> },
            ] as const
          ).map(({ id, label, icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                tab === id
                  ? 'bg-violet-600 text-white'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {icon}
              {label}
            </button>
          ))}
        </nav>

        <div className="w-24" /> {/* balance the header */}
      </header>

      {/* Tab content */}
      <main className="flex-1 overflow-hidden">
        <div className={`h-full ${tab === 'chat' ? 'block' : 'hidden'}`}>
          <ChatTab />
        </div>
        <div className={`h-full ${tab === 'metrics' ? 'block' : 'hidden'}`}>
          <MetricsTab />
        </div>
      </main>
    </div>
  )
}
