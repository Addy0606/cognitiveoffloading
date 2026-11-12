import { useEffect, useRef, useCallback } from 'react'
import { sendHeartbeat } from '../api/index.js'

/**
 * Active Cognitive Engagement Time (ACET) watchdog.
 * - Tracks window focus via visibilitychange
 * - Tracks last user activity (mouse / key / scroll / click)
 * - Accumulates active seconds per turn (pauses when idle > 60s or tab hidden)
 * - Sends /api/heartbeat every 25s when active
 *
 * Returns:
 *   consumeActiveDwell() → current accumulated seconds (clamped 5-180), then resets
 */
export function useACET({ currentConversationId, currentTopicId }) {
  const isWindowFocusedRef = useRef(!document.hidden)
  const lastActivityRef = useRef(Date.now())
  const activeDwellRef = useRef(0)
  const isSendingRef = useRef(false)   // set by App when waiting for LLM

  // activity recorder — passive listeners
  const recordActivity = useCallback(() => {
    lastActivityRef.current = Date.now()
  }, [])

  useEffect(() => {
    const opts = { passive: true }
    window.addEventListener('mousemove', recordActivity, opts)
    window.addEventListener('keydown', recordActivity, opts)
    window.addEventListener('scroll', recordActivity, opts)
    window.addEventListener('click', recordActivity, opts)
    return () => {
      window.removeEventListener('mousemove', recordActivity, opts)
      window.removeEventListener('keydown', recordActivity, opts)
      window.removeEventListener('scroll', recordActivity, opts)
      window.removeEventListener('click', recordActivity, opts)
    }
  }, [recordActivity])

  // visibility change
  useEffect(() => {
    const onVis = () => {
      isWindowFocusedRef.current = !document.hidden
      if (!document.hidden) lastActivityRef.current = Date.now()
    }
    document.addEventListener('visibilitychange', onVis)
    return () => document.removeEventListener('visibilitychange', onVis)
  }, [])

  // 1-second accumulator
  useEffect(() => {
    const timer = setInterval(() => {
      const isRecentlyActive = (Date.now() - lastActivityRef.current) <= 60000
      if (isWindowFocusedRef.current && isRecentlyActive && !isSendingRef.current) {
        activeDwellRef.current += 1
      }
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  // 25s heartbeat
  const convRef = useRef(currentConversationId)
  const topicRef = useRef(currentTopicId)
  useEffect(() => { convRef.current = currentConversationId }, [currentConversationId])
  useEffect(() => { topicRef.current = currentTopicId }, [currentTopicId])

  useEffect(() => {
    const hb = setInterval(() => {
      const isRecentlyActive = (Date.now() - lastActivityRef.current) <= 60000
      if (isWindowFocusedRef.current && isRecentlyActive && (topicRef.current || convRef.current)) {
        sendHeartbeat({
          topic_id: topicRef.current,
          conversation_id: convRef.current,
          active_seconds: 25.0
        })
      }
    }, 25000)
    return () => clearInterval(hb)
  }, [])

  const consumeActiveDwell = useCallback(() => {
    const val = Math.min(Math.max(activeDwellRef.current, 5), 180)
    activeDwellRef.current = 0
    return val
  }, [])

  return { consumeActiveDwell, isSendingRef }
}

