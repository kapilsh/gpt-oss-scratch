import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../types'

interface Props {
  message: Message
}

export function ChatMessage({ message }: Props) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className="max-w-[80%]">
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-violet-600 text-white rounded-tr-sm'
              : 'bg-surface border border-border text-slate-200 rounded-tl-sm'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <>
              {message.content ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    pre({ children }) {
                      return (
                        <pre className="bg-black/40 rounded-lg p-3 my-2 overflow-x-auto font-mono text-xs leading-relaxed">
                          {children}
                        </pre>
                      )
                    },
                    code({ children, className }) {
                      if (className?.startsWith('language-')) {
                        return <code className={className}>{children}</code>
                      }
                      return (
                        <code className="bg-black/40 rounded px-1 py-0.5 font-mono text-xs">
                          {children}
                        </code>
                      )
                    },
                    p({ children }) {
                      return <p className="mb-2 last:mb-0">{children}</p>
                    },
                    ul({ children }) {
                      return <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>
                    },
                    ol({ children }) {
                      return <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>
                    },
                    strong({ children }) {
                      return <strong className="font-semibold text-slate-100">{children}</strong>
                    },
                    blockquote({ children }) {
                      return (
                        <blockquote className="border-l-2 border-violet-500/40 pl-3 my-2 text-slate-400 italic">
                          {children}
                        </blockquote>
                      )
                    },
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              ) : (
                <span className="inline-flex gap-1 text-slate-500">
                  <span className="animate-bounce [animation-delay:0ms]">●</span>
                  <span className="animate-bounce [animation-delay:150ms]">●</span>
                  <span className="animate-bounce [animation-delay:300ms]">●</span>
                </span>
              )}
              {message.isStreaming && message.content && (
                <span className="inline-block w-1.5 h-3.5 bg-slate-400 animate-pulse ml-0.5 align-text-bottom" />
              )}
            </>
          )}
        </div>
        <p className="text-[10px] text-slate-600 mt-1 px-1">
          {message.timestamp.toLocaleTimeString()}
        </p>
      </div>
    </div>
  )
}
