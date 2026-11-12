import React, { useState, useEffect, useCallback, useRef } from 'react'
import Sidebar from './components/Sidebar/Sidebar.jsx'
import ChatArea from './components/Chat/ChatArea.jsx'
import RenameModal from './components/Modals/RenameModal.jsx'
import ConfidenceModal from './components/Modals/ConfidenceModal.jsx'
import TelemetryModal from './components/Modals/TelemetryModal.jsx'
import AnalyticsModal from './components/Modals/AnalyticsModal.jsx'
import {
  fetchConversations, fetchConversationMessages, deleteConversation,
  renameConversation, sendChatMessage, sendFeedback, fetchConcepts,
  fetchCheckin, resolveCheckin, fetchStatus
} from './api/index.js'
import { useACET } from './hooks/useACET.js'
import { useTheme } from './hooks/useTheme.js'

const MOBILE_QUERY = '(max-width: 820px)'
const isMobileView = () => window.matchMedia?.(MOBILE_QUERY).matches

export default function App() {
  // ── Theme + layout ──────────────────────────────────────────────
  const { theme, toggleTheme } = useTheme()
  const [sidebarOpen, setSidebarOpen] = useState(() => !isMobileView())
  const toggleSidebar = useCallback(() => setSidebarOpen(o => !o), [])
  const closeSidebarOnMobile = useCallback(() => { if (isMobileView()) setSidebarOpen(false) }, [])

  // ── Core state ──────────────────────────────────────────────────
  const [conversations, setConversations] = useState([])
  const [currentConvId, setCurrentConvId] = useState(null)
  const [currentTopicId, setCurrentTopicId] = useState(null)
  const [messages, setMessages] = useState([])
  const [topicInfo, setTopicInfo] = useState(null)
  const [isSending, setIsSending] = useState(false)
  const [providerLabel, setProviderLabel] = useState('OLLAMA')

  // ── Mastery sidebar state ───────────────────────────────────────
  const [mastery, setMastery] = useState({ masteredCount: 0, practiced: 0, avgConf: '--', pct: 0 })

  // ── Retention banner ────────────────────────────────────────────
  const [checkinData, setCheckinData] = useState(null)
  const [showRetention, setShowRetention] = useState(false)

  // ── Modals ──────────────────────────────────────────────────────
  const [renameModal, setRenameModal] = useState({ open: false, convId: null, currentTitle: '' })
  const [confidenceModal, setConfidenceModal] = useState({ open: false, messageId: null, conceptId: null })
  const [telemetryModal, setTelemetryModal] = useState(false)
  const [analyticsModal, setAnalyticsModal] = useState(false)
  const [telemetryData, setTelemetryData] = useState(null)

  // ── ACET watchdog ───────────────────────────────────────────────
  const { consumeActiveDwell, isSendingRef } = useACET({ currentConversationId: currentConvId, currentTopicId })
  useEffect(() => { isSendingRef.current = isSending }, [isSending, isSendingRef])

  // ── Init ─────────────────────────────────────────────────────────
  useEffect(() => {
    loadConversations({ autoOpen: true })
    loadMastery()
    loadStatus()
    checkRetention()
  }, []) // eslint-disable-line

  // ── Load conversations ───────────────────────────────────────────
  const currentConvIdRef = useRef(currentConvId)
  useEffect(() => { currentConvIdRef.current = currentConvId }, [currentConvId])

  // Navigation counter: bumped on every New Chat / conversation open. Async loads that
  // finish after a newer navigation are discarded, so a slow startup auto-open can't
  // overwrite a chat the user has already switched to (or a fresh New Chat).
  const navSeqRef = useRef(0)

  const loadConversations = useCallback(async ({ autoOpen = false } = {}) => {
    const seqAtStart = navSeqRef.current
    try {
      const data = await fetchConversations()
      const convs = data.conversations || []
      setConversations(convs)
      if (autoOpen && !currentConvIdRef.current && navSeqRef.current === seqAtStart) {
        const savedId = localStorage.getItem('dsa_active_conv_id')
        const target = convs.find(c => c.id === savedId) || convs[0]
        if (target) loadMessages(target.id)
      }
    } catch (e) { console.error(e) }
  }, []) // eslint-disable-line

  const loadMessages = useCallback(async (convId) => {
    const seq = ++navSeqRef.current
    try {
      const data = await fetchConversationMessages(convId)
      if (seq !== navSeqRef.current) return // user navigated elsewhere meanwhile
      currentConvIdRef.current = convId
      setCurrentConvId(convId)
      localStorage.setItem('dsa_active_conv_id', convId)
      const conv = data.conversation
      const topic = data.topic
      const msgs = data.messages || []
      setCurrentTopicId(conv.topic_id || (topic ? topic.concept_id : null))
      setTopicInfo(topic
        ? { title: conv.title || 'DSA Chat', category: topic.category || 'DSA', concept_name: topic.concept_name }
        : { title: conv.title || 'DSA Chat', category: null, concept_name: null })
      setMessages(msgs.map((m, idx) => ({
        ...m,
        meta: m.meta || {},
        show_feedback: Boolean(
          m.meta?.show_feedback ||
          m.feedback_rating != null ||
          m.feedback_label != null ||
          idx === msgs.length - 1 ||
          (idx === msgs.length - 2 && msgs[msgs.length - 1]?.role === 'student')
        )
      })))
    } catch (e) { console.error('Failed to load conversation', e) }
  }, [])

  const startNewChat = useCallback(() => {
    navSeqRef.current++
    currentConvIdRef.current = null
    setCurrentConvId(null)
    setCurrentTopicId(null)
    setMessages([])
    setTopicInfo(null)
    localStorage.removeItem('dsa_active_conv_id')
    checkRetention()
  }, []) // eslint-disable-line

  const checkRetention = useCallback(async () => {
    if (sessionStorage.getItem('dsa_checkin_dismissed') === 'true') { setShowRetention(false); return }
    try {
      const data = await fetchCheckin()
      if (data.checkin?.concept_name) { setCheckinData(data.checkin); setShowRetention(true) }
      else setShowRetention(false)
    } catch (_) { setShowRetention(false) }
  }, [])

  const loadMastery = useCallback(async () => {
    try {
      const data = await fetchConcepts()
      const concepts = data.concepts || []
      const practiced = concepts.filter(c => (c.total_turns || 0) > 0 || (c.confidence_score || 0) > 0)
      const mastered = concepts.filter(c => (c.confidence_score || 0) >= 8)
      const pct = concepts.length > 0 ? Math.round((mastered.length / concepts.length) * 100) : 0
      const scores = practiced.map(c => c.confidence_score).filter(s => s > 0)
      const avg = scores.length > 0 ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) : '--'
      setMastery({ masteredCount: mastered.length, practiced: practiced.length, avgConf: avg, pct })
    } catch (e) { console.error(e) }
  }, [])

  const loadStatus = useCallback(async () => {
    try {
      const data = await fetchStatus()
      if (data.llm_provider) setProviderLabel(`${data.llm_provider.toUpperCase()} (${data.llm_model || 'qwen2.5:7b'})`)
    } catch (_) {}
  }, [])

  // ── Delete / rename ──────────────────────────────────────────────
  const handleDeleteConv = useCallback(async (convId) => {
    await deleteConversation(convId)
    const wasActive = currentConvIdRef.current === convId
    if (wasActive) startNewChat()
    // Deleting the open chat falls through to the most recent remaining one (previous behaviour)
    await Promise.all([loadConversations({ autoOpen: wasActive }), loadMastery()])
  }, [startNewChat, loadConversations, loadMastery])

  const handleRename = useCallback(async (convId, newTitle) => {
    await renameConversation(convId, newTitle)
    if (currentConvIdRef.current === convId) setTopicInfo(prev => ({ ...prev, title: newTitle }))
    setRenameModal({ open: false, convId: null, currentTitle: '' })
    await loadConversations()
  }, [loadConversations])

  // ── Send message ─────────────────────────────────────────────────
  const handleSendMessage = useCallback(async (text) => {
    if (!text || isSending) return
    const activeDwell = consumeActiveDwell()
    setIsSending(true)

    const typingId = `typing_${Date.now()}`
    setMessages(prev => [
      ...prev,
      { id: `stu_${Date.now()}`, role: 'student', content: text, meta: {} },
      { id: typingId, role: '__typing__', meta: {} }
    ])

    try {
      const data = await sendChatMessage({
        message: text,
        conversation_id: currentConvIdRef.current,
        topic_id: currentTopicId,
        active_dwell_seconds: activeDwell
      })

      setMessages(prev => {
        const filtered = prev.filter(m => m.id !== typingId)
        if (data.error) return [...filtered, { id: `sys_${Date.now()}`, role: '__system__', content: `⚠️ Error: ${data.error}`, meta: {} }]
        const result = [...filtered]
        if (data.topic_switched && data.old_topic_name && data.old_topic_id) {
          result.push({ id: data.transition_msg_id || `tt_${Date.now()}`, role: 'topic_transition', content: '', meta: { old_topic_id: data.old_topic_id, old_topic_name: data.old_topic_name, new_topic_name: data.topic_name }, feedback_rating: null, feedback_label: null })
        }
        result.push({ id: data.message_id || `msg_${Date.now()}`, role: 'tutor', content: data.response, meta: {}, feedback_rating: null, feedback_label: null, show_feedback: Boolean(data.show_feedback) })
        return result
      })

      if (!data.error) {
        if (!currentConvIdRef.current && data.conversation_id) {
          currentConvIdRef.current = data.conversation_id
          setCurrentConvId(data.conversation_id)
        }
        if (data.topic_id) setCurrentTopicId(data.topic_id)
        if (data.title || data.topic_name) {
          setTopicInfo(prev => ({
            ...prev,
            title: data.title || prev?.title || 'DSA Chat',
            category: data.topic_name || prev?.category,
            concept_name: data.topic_name || prev?.concept_name,
          }))
        }
        setTelemetryData(data)
        await Promise.all([loadConversations(), loadMastery()])
      }
    } catch (err) {
      setMessages(prev => {
        const filtered = prev.filter(m => m.id !== typingId)
        return [...filtered, { id: `sys_${Date.now()}`, role: '__system__', content: `⚠️ Connection error: ${err.message}`, meta: {} }]
      })
    } finally {
      setIsSending(false)
    }
  }, [isSending, currentTopicId, consumeActiveDwell, loadConversations, loadMastery])

  // ── Feedback ─────────────────────────────────────────────────────
  const handleFeedback = useCallback(async ({ messageId, rating, label, conceptId }) => {
    try {
      const data = await sendFeedback({ message_id: messageId, rating, label, concept_id: conceptId, conversation_id: currentConvIdRef.current })
      setMessages(prev => prev.map(m => m.id !== messageId ? m : { ...m, feedback_rating: rating ?? m.feedback_rating, feedback_label: label ?? m.feedback_label }))
      const result = []
      if (label === 'confused') result.push({ id: `sys_${Date.now()}`, role: '__system__', content: '💡 <em>Feedback noted: The tutor is diagnosing this step to clear up confusion.</em>', meta: {}, isHtml: true })
      else if (label === 'clear') result.push({ id: `sys_${Date.now()}`, role: '__system__', content: '👍 <em>Feedback noted: Concept clarity recorded! The tutor will test deeper comprehension.</em>', meta: {}, isHtml: true })
      if (data?.adaptive_intervention?.prompt) result.push({ id: data.adaptive_intervention.message_id || `adp_${Date.now()}`, role: 'tutor', content: data.adaptive_intervention.prompt, meta: { is_adaptive: true }, show_feedback: false })
      if (result.length > 0) setMessages(prev => [...prev, ...result])
      await loadMastery()
    } catch (e) { console.error('Feedback error', e) }
  }, [loadMastery])

  // ── Confidence modal save ─────────────────────────────────────────
  const handleConfidenceSave = useCallback(async ({ messageId, conceptId, rating, notes }) => {
    let adaptiveData = null
    if (conceptId) {
      await resolveCheckin({ concept_id: conceptId, confidence_score: rating, notes })
      try { adaptiveData = await sendFeedback({ message_id: messageId, rating, concept_id: conceptId, conversation_id: currentConvIdRef.current }) } catch (_) {}
    } else if (messageId) {
      try { adaptiveData = await sendFeedback({ message_id: messageId, rating, concept_id: conceptId, conversation_id: currentConvIdRef.current }) } catch (_) {}
    }
    if (messageId) setMessages(prev => prev.map(m => m.id === messageId ? { ...m, feedback_rating: rating } : m))
    const result = []
    if (rating <= 4) result.push({ id: `sys_${Date.now()}`, role: '__system__', content: `💡 <em>Rated ${rating}/10: The tutor is diagnosing this step to clear up confusion.</em>`, meta: {}, isHtml: true })
    else if (rating >= 8) result.push({ id: `sys_${Date.now()}`, role: '__system__', content: `⭐ <em>Rated ${rating}/10: High confidence recorded! Ready for deeper algorithmic reasoning.</em>`, meta: {}, isHtml: true })
    if (adaptiveData?.adaptive_intervention?.prompt) result.push({ id: adaptiveData.adaptive_intervention.message_id || `adp_${Date.now()}`, role: 'tutor', content: adaptiveData.adaptive_intervention.prompt, meta: { is_adaptive: true }, show_feedback: false })
    if (result.length > 0) setMessages(prev => [...prev, ...result])
    sessionStorage.setItem('dsa_checkin_dismissed', 'true')
    setShowRetention(false)
    setConfidenceModal({ open: false, messageId: null, conceptId: null })
    await loadMastery()
  }, [loadMastery])

  // ── Retention handlers ────────────────────────────────────────────
  const handleRetentionTest = useCallback(() => {
    if (!checkinData) return
    setShowRetention(false)
    sessionStorage.setItem('dsa_checkin_dismissed', 'true')
    handleSendMessage(`Can you ask me a quick 1-minute question to test my understanding of ${checkinData.concept_name}?`)
  }, [checkinData, handleSendMessage])

  const handleRetentionRate = useCallback(() => {
    if (!checkinData) return
    setConfidenceModal({ open: true, messageId: null, conceptId: checkinData.concept_id })
  }, [checkinData])

  const handleRetentionDismiss = useCallback(() => {
    setShowRetention(false)
    sessionStorage.setItem('dsa_checkin_dismissed', 'true')
  }, [])

  const startNewChatWithTopic = useCallback((conceptId, conceptName) => {
    startNewChat()
    setCurrentTopicId(conceptId)
    setTopicInfo({ title: conceptName, category: 'DSA Concept', concept_name: conceptName })
    setAnalyticsModal(false)
  }, [startNewChat])

  return (
    <div className={`app-layout${sidebarOpen ? '' : ' sidebar-collapsed'}`}>
      <Sidebar
        conversations={conversations}
        currentConvId={currentConvId}
        mastery={mastery}
        onNewChat={() => { startNewChat(); closeSidebarOnMobile() }}
        onSelectConv={(id) => { loadMessages(id); closeSidebarOnMobile() }}
        onDeleteConv={handleDeleteConv}
        onRenameConv={(convId, title) => setRenameModal({ open: true, convId, currentTitle: title })}
        onOpenTelemetry={() => setTelemetryModal(true)}
        onOpenAnalytics={() => setAnalyticsModal(true)}
        isOpen={sidebarOpen}
        onToggleSidebar={toggleSidebar}
      />
      {sidebarOpen && <div className="sidebar-scrim" onClick={() => setSidebarOpen(false)} aria-hidden="true" />}
      <ChatArea
        theme={theme}
        onToggleTheme={toggleTheme}
        isSidebarOpen={sidebarOpen}
        onToggleSidebar={toggleSidebar}
        messages={messages}
        topicInfo={topicInfo}
        currentConvId={currentConvId}
        currentTopicId={currentTopicId}
        isSending={isSending}
        showRetention={showRetention}
        checkinData={checkinData}
        onSendMessage={handleSendMessage}
        onFeedback={handleFeedback}
        onOpenConfidence={(msgId, cid) => setConfidenceModal({ open: true, messageId: msgId, conceptId: cid })}
        onRenameHeader={() => {
          if (!currentConvId) return
          setRenameModal({ open: true, convId: currentConvId, currentTitle: topicInfo?.title || 'DSA Chat' })
        }}
        onRetentionTest={handleRetentionTest}
        onRetentionRate={handleRetentionRate}
        onRetentionDismiss={handleRetentionDismiss}
      />

      {renameModal.open && <RenameModal convId={renameModal.convId} currentTitle={renameModal.currentTitle} onSave={handleRename} onClose={() => setRenameModal({ open: false })} />}
      {confidenceModal.open && <ConfidenceModal messageId={confidenceModal.messageId} conceptId={confidenceModal.conceptId} onSave={handleConfidenceSave} onClose={() => setConfidenceModal({ open: false })} />}
      {telemetryModal && <TelemetryModal data={telemetryData} onClose={() => setTelemetryModal(false)} />}
      {analyticsModal && <AnalyticsModal onClose={() => setAnalyticsModal(false)} onPractice={startNewChatWithTopic} />}
    </div>
  )
}
