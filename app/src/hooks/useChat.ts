import { useCallback, useState } from 'react'
import type { ChatParams, Message } from '../types'

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)

  const updateAssistant = useCallback((id: string, content: string, done: boolean) => {
    setMessages(prev =>
      prev.map(m =>
        m.id === id ? { ...m, content, isStreaming: !done } : m,
      ),
    )
  }, [])

  const sendMessage = useCallback(
    async (userContent: string, params: ChatParams) => {
      if (isStreaming) return

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: 'user',
        content: userContent,
        reasoning: '',
        isThinking: false,
        isStreaming: false,
        timestamp: new Date(),
      }

      const assistantId = crypto.randomUUID()
      const assistantMsg: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        reasoning: '',
        isThinking: false,
        isStreaming: true,
        timestamp: new Date(),
      }

      setMessages(prev => [...prev, userMsg, assistantMsg])
      setIsStreaming(true)

      try {
        const body = {
          model: 'gpt-oss-20b',
          messages: [
            ...(params.systemPrompt
              ? [{ role: 'system', content: params.systemPrompt }]
              : []),
            ...messages
              .filter(m => !m.isStreaming)
              .map(m => ({ role: m.role, content: m.content })),
            { role: 'user', content: userContent },
          ],
          temperature: params.temperature,
          top_p: params.topP,
          max_tokens: params.maxTokens,
          stream: true,
        }

        const res = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })

        if (!res.ok || !res.body) {
          throw new Error(`Server error: ${res.status}`)
        }

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let accumulated = ''
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            const data = line.slice(6).trim()
            if (data === '[DONE]') break
            try {
              const parsed = JSON.parse(data)
              const delta = parsed.choices?.[0]?.delta?.content ?? ''
              accumulated += delta
              updateAssistant(assistantId, accumulated, false)
            } catch {
              // malformed chunk — skip
            }
          }
        }

        updateAssistant(assistantId, accumulated, true)
      } catch (err) {
        updateAssistant(
          assistantId,
          `Error: ${err instanceof Error ? err.message : String(err)}`,
          true,
        )
      } finally {
        setIsStreaming(false)
      }
    },
    [isStreaming, messages, updateAssistant],
  )

  const clearMessages = useCallback(() => setMessages([]), [])

  return { messages, isStreaming, sendMessage, clearMessages }
}
