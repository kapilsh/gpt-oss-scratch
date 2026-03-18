import type { MetricsSnapshot } from '../types'

// Parses Prometheus text exposition format into a flat key→value map.
// Keys include label sets e.g. `gptoss_requests_total{status="success"}`.
function parsePrometheusText(text: string): Record<string, number> {
  const result: Record<string, number> = {}
  for (const line of text.split('\n')) {
    if (line.startsWith('#') || !line.trim()) continue
    const m = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(-?[\d.]+(?:e[+-]?\d+)?|NaN|[+-]?Inf)/)
    if (!m) continue
    const key = m[1] + (m[2] ?? '')
    const value = parseFloat(m[3])
    result[key] = isNaN(value) ? 0 : value
  }
  return result
}

function get(m: Record<string, number>, key: string): number {
  return m[key] ?? 0
}

export function buildSnapshot(
  text: string,
  prev: MetricsSnapshot | null,
): MetricsSnapshot {
  const m = parsePrometheusText(text)
  const now = Date.now()

  const promptTotal = get(m, 'gptoss_prompt_tokens_total_total') ||
                      get(m, 'gptoss_prompt_tokens_total')
  const completionTotal = get(m, 'gptoss_completion_tokens_total_total') ||
                          get(m, 'gptoss_completion_tokens_total')

  // Average latency from histogram sum/count
  const latencySum = get(m, 'gptoss_request_latency_seconds_sum')
  const latencyCount = get(m, 'gptoss_request_latency_seconds_count')
  const latencyAvg = latencyCount > 0 ? latencySum / latencyCount : 0

  const ttftSum = get(m, 'gptoss_ttft_seconds_sum')
  const ttftCount = get(m, 'gptoss_ttft_seconds_count')
  const ttftAvg = ttftCount > 0 ? ttftSum / ttftCount : 0

  // Token throughput rates from counter deltas
  let promptTokensRate = 0
  let completionTokensRate = 0
  if (prev) {
    const dt = (now - prev.timestamp) / 1000
    if (dt > 0) {
      promptTokensRate = Math.max(0, (promptTotal - prev.promptTokensTotal) / dt)
      completionTokensRate = Math.max(0, (completionTotal - prev.completionTokensTotal) / dt)
    }
  }

  return {
    timestamp: now,
    requestsInFlight: get(m, 'gptoss_requests_in_flight'),
    requestsSuccess: get(m, 'gptoss_requests_total{status="success"}'),
    requestsError: get(m, 'gptoss_requests_total{status="error"}'),
    promptTokensTotal: promptTotal,
    completionTokensTotal: completionTotal,
    latencyAvg,
    ttftAvg,
    promptTokensRate,
    completionTokensRate,
  }
}
