export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  reasoning: string       // content inside <think>...</think>
  isThinking: boolean     // still inside an open <think> tag
  isStreaming: boolean
  timestamp: Date
}

export interface ChatParams {
  temperature: number
  topP: number
  maxTokens: number
  systemPrompt: string
}

export interface MetricsSnapshot {
  timestamp: number
  requestsInFlight: number
  requestsSuccess: number
  requestsError: number
  promptTokensTotal: number
  completionTokensTotal: number
  latencyAvg: number       // seconds, computed from sum/count
  ttftAvg: number          // seconds
  // rates (delta from previous snapshot)
  promptTokensRate: number   // tokens/sec
  completionTokensRate: number
}

export type Tab = 'chat' | 'metrics'
