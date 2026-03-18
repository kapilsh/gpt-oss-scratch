import { useEffect, useRef, useState } from 'react'
import type { MetricsSnapshot } from '../types'
import { buildSnapshot } from '../utils/parseMetrics'

const MAX_HISTORY = 120   // ~6 minutes at 3s interval

export function useMetrics(intervalMs = 3000) {
  const [history, setHistory] = useState<MetricsSnapshot[]>([])
  const [error, setError] = useState<string | null>(null)
  const prevRef = useRef<MetricsSnapshot | null>(null)

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      try {
        const res = await fetch('/metrics')
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const text = await res.text()
        if (cancelled) return
        const snapshot = buildSnapshot(text, prevRef.current)
        prevRef.current = snapshot
        setHistory(h => [...h.slice(-(MAX_HISTORY - 1)), snapshot])
        setError(null)
      } catch (e) {
        if (!cancelled) setError(String(e))
      }
    }

    poll()
    const id = setInterval(poll, intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [intervalMs])

  return { history, error }
}
