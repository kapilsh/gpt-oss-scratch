import { useEffect, useRef, useState } from 'react'
import { Send, Trash2, Settings2, ChevronDown, ChevronUp } from 'lucide-react'
import type { ChatParams } from '../types'
import { useChat } from '../hooks/useChat'
import { ChatMessage } from './ChatMessage'

const DEFAULT_PARAMS: ChatParams = {
  temperature: 0.7,
  topP: 0.9,
  maxTokens: 4096,
  systemPrompt: '',
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  display,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  display?: (v: number) => string
}) {
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-400 mb-1">
        <span>{label}</span>
        <span className="font-mono text-slate-300">{display ? display(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full accent-violet-500 bg-border cursor-pointer"
      />
    </div>
  )
}

export function ChatTab() {
  const { messages, isStreaming, sendMessage, clearMessages } = useChat()
  const [input, setInput] = useState('')
  const [params, setParams] = useState<ChatParams>(DEFAULT_PARAMS)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    const text = input.trim()
    if (!text || isStreaming) return
    setInput('')
    sendMessage(text, params)
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`
  }

  return (
    <div className="flex flex-col h-full">
      {/* Settings panel */}
      <div className="border-b border-border flex-shrink-0">
        <button
          onClick={() => setSettingsOpen(o => !o)}
          className="w-full flex items-center gap-2 px-4 py-2.5 text-xs text-slate-400 hover:text-slate-300 transition-colors"
        >
          <Settings2 size={13} />
          <span>Parameters</span>
          <span className="ml-auto text-slate-600 font-mono">
            T={params.temperature} · top_p={params.topP} · max={params.maxTokens}
          </span>
          {settingsOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        {settingsOpen && (
          <div className="px-4 pb-4 grid grid-cols-3 gap-6 border-t border-border pt-3">
            <Slider
              label="Temperature"
              value={params.temperature}
              min={0}
              max={2}
              step={0.05}
              onChange={v => setParams(p => ({ ...p, temperature: v }))}
              display={v => v.toFixed(2)}
            />
            <Slider
              label="Top P"
              value={params.topP}
              min={0}
              max={1}
              step={0.05}
              onChange={v => setParams(p => ({ ...p, topP: v }))}
              display={v => v.toFixed(2)}
            />
            <Slider
              label="Max Tokens"
              value={params.maxTokens}
              min={64}
              max={8192}
              step={64}
              onChange={v => setParams(p => ({ ...p, maxTokens: v }))}
            />
            <div className="col-span-3">
              <label className="text-xs text-slate-400 block mb-1">System Prompt</label>
              <textarea
                value={params.systemPrompt}
                onChange={e => setParams(p => ({ ...p, systemPrompt: e.target.value }))}
                placeholder="Optional system prompt..."
                rows={2}
                className="w-full bg-black/30 border border-border rounded-lg px-3 py-2 text-xs text-slate-300 placeholder-slate-600 resize-none focus:outline-none focus:border-violet-500/50"
              />
            </div>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center text-slate-600">
            <div className="text-4xl mb-3">◈</div>
            <p className="text-sm">GPT-OSS 20B</p>
            <p className="text-xs mt-1">Start a conversation below</p>
          </div>
        )}
        {messages.map(msg => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="border-t border-border px-4 py-3 flex-shrink-0">
        <div className="flex items-end gap-2">
          <button
            onClick={clearMessages}
            title="Clear conversation"
            className="mb-1 p-1.5 text-slate-600 hover:text-slate-400 transition-colors flex-shrink-0"
          >
            <Trash2 size={15} />
          </button>
          <div className="flex-1">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Message GPT-OSS… (Enter to send, Shift+Enter for newline)"
              rows={1}
              disabled={isStreaming}
              className="w-full bg-surface border border-border rounded-xl px-4 py-2.5 text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none focus:border-violet-500/50 transition-colors disabled:opacity-50"
              style={{ minHeight: '44px' }}
            />
          </div>
          <button
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
            className="mb-1 p-2 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:bg-slate-700 disabled:text-slate-500 text-white transition-colors flex-shrink-0"
          >
            <Send size={15} />
          </button>
        </div>
        <p className="text-[10px] text-slate-700 mt-1.5 text-center">
          {isStreaming ? 'Generating…' : 'Ready'} · vLLM · PagedAttention
        </p>
      </div>
    </div>
  )
}
