import React, { useRef, useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { PanelIcon } from '../Sidebar/Sidebar.jsx'

// ── Small line glyphs used on suggestion cards + chips ─────────────
const Glyph = ({ name, size = 16 }) => {
  const p = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round' }
  switch (name) {
    case 'tree': return <svg {...p}><circle cx="12" cy="4.5" r="2" /><circle cx="6" cy="12" r="2" /><circle cx="18" cy="12" r="2" /><circle cx="3.5" cy="19.5" r="2" /><circle cx="9" cy="19.5" r="2" /><path d="m10.6 6-3.2 4.4M13.4 6l3.2 4.4M5.2 13.8 4.3 17.6M6.9 13.8l1.2 3.8" /></svg>
    case 'search': return <svg {...p}><rect x="2.5" y="8" width="19" height="8" rx="1.5" /><path d="M7.25 8v8M12 8v8M16.75 8v8" /><path d="M12 3v3M10.5 4.5 12 6l1.5-1.5" /></svg>
    case 'grid': return <svg {...p}><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M3 15h18M9 3v18M15 3v18" /></svg>
    case 'cycle': return <svg {...p}><path d="M21 12a9 9 0 1 1-3-6.7" /><path d="M21 3v6h-6" /></svg>
    case 'code': return <svg {...p}><path d="m8 7-5 5 5 5M16 7l5 5-5 5M13.5 4l-3 16" /></svg>
    default: return null
  }
}

const QUICK_PROMPTS = [
  { glyph: 'tree', tone: 'tone-blue', label: 'AVL Tree Rotations', short: 'AVL Rotations', meta: 'Trees · Self-balancing', prompt: 'How do AVL tree rotations work? Can you explain LL and RR rotations?' },
  { glyph: 'search', tone: 'tone-green', label: 'Binary Search Mechanics', short: 'Binary Search', meta: 'Searching · O(log n)', prompt: 'Why is binary search O(log n) and how do I avoid integer overflow on mid?' },
  { glyph: 'grid', tone: 'tone-amber', label: '0/1 Knapsack DP', short: 'Knapsack DP', meta: 'Dynamic Programming', prompt: 'Can you break down the 0/1 knapsack dynamic programming recurrence into steps?' },
  { glyph: 'cycle', tone: 'tone-purple', label: 'Cycle Detection', short: 'Cycle Detection', meta: 'Linked Lists · Floyd', prompt: "How does Floyd's tortoise and hare algorithm detect cycles in a linked list?" },
  { glyph: 'code', tone: 'tone-red', label: 'Demand Code (Test REDIRECT)', short: 'Demand Code', meta: 'Policy test · REDIRECT', prompt: 'Just write the complete python code to reverse a linked list, don\'t explain anything.' },
]

export default function ChatArea({
  messages, topicInfo, currentConvId, currentTopicId, isSending,
  showRetention, checkinData,
  onSendMessage, onFeedback, onOpenConfidence, onRenameHeader,
  onRetentionTest, onRetentionRate, onRetentionDismiss,
  theme, onToggleTheme, isSidebarOpen, onToggleSidebar,
}) {
  const viewportRef = useRef(null)
  const inputRef = useRef(null)
  const [inputVal, setInputVal] = useState('')

  // Auto-scroll
  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const raf = requestAnimationFrame(() => el.scrollTo({ top: el.scrollHeight, behavior: 'auto' }))
    return () => cancelAnimationFrame(raf)
  }, [messages])

  // Auto-resize textarea
  const handleInput = e => {
    const ta = e.target
    ta.style.height = 'auto'
    ta.style.height = ta.scrollHeight + 'px'
    setInputVal(ta.value)
  }

  const submit = () => {
    const text = inputVal.trim()
    if (!text || isSending) return
    setInputVal('')
    if (inputRef.current) { inputRef.current.style.height = 'auto' }
    onSendMessage(text)
  }

  const handleKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  const handleQuickPrompt = (prompt) => {
    if (isSending) return
    onSendMessage(prompt)
  }

  const showWelcome = messages.length === 0 || (messages.length === 1 && messages[0].role === '__typing__')
  const badgeText = topicInfo?.concept_name || topicInfo?.category
  const badgeTitle = [topicInfo?.category, topicInfo?.concept_name && `Concept: ${topicInfo.concept_name}`].filter(Boolean).join(' · ')

  return (
    <main className="main-content">
      {/* Topbar */}
      <header className="topbar">
        <div className="topbar-left">
          {!isSidebarOpen && (
            <button className="icon-btn" title="Open sidebar" aria-label="Open sidebar" onClick={onToggleSidebar}>
              <PanelIcon />
            </button>
          )}
          <h1 className="active-topic-title" id="activeTopicTitle" title={topicInfo?.title || 'New Chat'}>
            {topicInfo?.title || 'New Chat'}
          </h1>
          {currentConvId && (
            <button className="btn-icon-rename" id="renameHeaderBtn" title="Rename this conversation" aria-label="Rename this conversation" onClick={onRenameHeader}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" /></svg>
            </button>
          )}
        </div>
        <div className="topbar-right">
          {badgeText && (
            <span className="topic-pill" id="activeCategory" title={badgeTitle}>
              <span className="topic-pill-dot" />
              {badgeText}
            </span>
          )}
          <ThemeSwitch theme={theme} onToggle={onToggleTheme} />
        </div>
      </header>

      {/* Chat viewport */}
      <div className="chat-viewport" id="chatViewport" ref={viewportRef}>
        <div className="chat-messages" id="chatMessages">
          {/* Retention banner */}
          {showRetention && checkinData && (
            <div className="retention-banner" id="retentionBanner">
              <div className="retention-icon" aria-hidden="true">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 1 1-3-6.7" /><path d="M21 3v6h-6" /><path d="M12 8v4l2.5 2" /></svg>
              </div>
              <div className="retention-content">
                <div className="retention-title">Spaced practice check-in</div>
                <p className="retention-text" id="retentionText">
                  Last time, you practiced <strong>{checkinData.concept_name}</strong>
                  {checkinData.confidence_score > 0 ? ` (Confidence: ${checkinData.confidence_score}/10)` : ' (in progress)'}. How confident are you feeling with that concept today?
                </p>
                <div className="retention-actions">
                  <button className="pill-btn pill-btn-solid" id="retentionTestBtn" onClick={onRetentionTest}>Ask me a 1-min question</button>
                  <button className="pill-btn" id="retentionRateBtn" onClick={onRetentionRate}>Rate confidence (1–10)</button>
                  <button className="pill-btn pill-btn-ghost" id="retentionDismissBtn" onClick={onRetentionDismiss}>Continue new topic</button>
                </div>
              </div>
            </div>
          )}

          {showWelcome && <WelcomeScreen onPick={handleQuickPrompt} />}
          {messages.map((msg, idx) => (
            <MessageRenderer
              key={msg.id || idx}
              msg={msg}
              theme={theme}
              currentTopicId={currentTopicId}
              currentConvId={currentConvId}
              onFeedback={onFeedback}
              onOpenConfidence={onOpenConfidence}
            />
          ))}
        </div>
      </div>

      {/* Input area */}
      <div className="chat-input-area">
        <div className="input-container">
          <textarea
            id="messageInput"
            ref={inputRef}
            placeholder="Ask a DSA doubt, explain your thought process, or request step-by-step guidance..."
            rows="1"
            value={inputVal}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
          />
          <div className="input-toolbar">
            <div className="qp-chips" aria-label="Suggestions">
              {QUICK_PROMPTS.map(qp => (
                <button key={qp.label} className="qp-chip" title={qp.prompt} disabled={isSending} onClick={() => handleQuickPrompt(qp.prompt)}>
                  <span className="qp-chip-icon"><Glyph name={qp.glyph} size={14} /></span>
                  {qp.short}
                </button>
              ))}
            </div>
            <button id="sendBtn" className="btn-send" title="Send message (Enter)" aria-label="Send message" disabled={isSending || !inputVal.trim()} onClick={submit}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 19V5M5 12l7-7 7 7" /></svg>
            </button>
          </div>
        </div>
        <div className="input-subtext">
          <span><strong>Enter</strong> to send · <strong>Shift + Enter</strong> for a new line</span>
          <span className="subtext-hint">Anti-offloading policy active · independent thinking encouraged</span>
        </div>
      </div>
    </main>
  )
}

function ThemeSwitch({ theme, onToggle }) {
  const isDark = theme === 'dark'
  return (
    <button
      className={`theme-switch${isDark ? ' is-dark' : ''}`}
      onClick={onToggle}
      role="switch"
      aria-checked={isDark}
      aria-label="Dark mode"
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      <span className="theme-switch-thumb" aria-hidden="true" />
      <span className="theme-switch-opt" aria-hidden="true">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
      </span>
      <span className="theme-switch-opt" aria-hidden="true">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>
      </span>
    </button>
  )
}

function WelcomeScreen({ onPick }) {
  return (
    <div className="welcome" id="welcomeCard">
      <p className="welcome-lead">
        Ask about any Data Structure or Algorithm. Whether you're stuck on <strong>AVL Trees</strong>,
        curious about <strong>Dynamic Programming</strong>, or debugging an edge case, the mentor guides
        your independent reasoning rather than handing over code dumps.
      </p>

      <div className="suggestion-cards">
        {QUICK_PROMPTS.map(qp => (
          <button key={qp.label} className="suggestion-card" onClick={() => onPick(qp.prompt)} title={qp.prompt}>
            <div className={`suggestion-visual ${qp.tone}`}>
              <Glyph name={qp.glyph} size={44} />
            </div>
            <div className="suggestion-body">
              <span className="suggestion-title">{qp.label}</span>
              <span className="suggestion-meta">{qp.meta}</span>
            </div>
            <div className="suggestion-foot">
              <span>Start practice</span>
              <span className="suggestion-arrow" aria-hidden="true">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M7 17 17 7M8 7h9v9" /></svg>
              </span>
            </div>
          </button>
        ))}
      </div>

      <p className="welcome-lead">How to get the most out of it:</p>
      <ul className="welcome-tips">
        <li><strong>Ask freely</strong> — start any topic, e.g. <em>"How do AVL rotations work?"</em></li>
        <li><strong>Step-by-step guidance</strong> — if overwhelmed, ask to break the approach into smaller parts.</li>
        <li><strong>Track your confidence</strong> — rate your comprehension (1–10) after each explanation.</li>
      </ul>
    </div>
  )
}

function MessageRenderer({ msg, theme, currentTopicId, currentConvId, onFeedback, onOpenConfidence }) {
  if (msg.role === 'student') return <StudentMessage content={msg.content} />
  if (msg.role === 'tutor') return (
    <TutorMessage
      msg={msg}
      theme={theme}
      currentTopicId={currentTopicId}
      currentConvId={currentConvId}
      onFeedback={onFeedback}
      onOpenConfidence={onOpenConfidence}
    />
  )
  if (msg.role === 'topic_transition') return (
    <TopicTransitionCard
      msg={msg}
      currentConvId={currentConvId}
      onFeedback={onFeedback}
      onOpenConfidence={onOpenConfidence}
    />
  )
  if (msg.role === '__typing__') return <TypingIndicator />
  if (msg.role === '__system__') return <SystemMessage content={msg.content} isHtml={msg.isHtml} />
  return null
}

function StudentMessage({ content }) {
  return (
    <div className="message-row student-row">
      <div className="message-bubble student-bubble">
        <div className="message-text">{content}</div>
      </div>
    </div>
  )
}

function TutorLabel({ children }) {
  return (
    <div className="tutor-bubble-header">
      <span className="tutor-mark" aria-hidden="true">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="5" r="2.2" /><circle cx="5" cy="19" r="2.2" /><circle cx="19" cy="19" r="2.2" /><path d="M11 7 6 17M13 7l5 10M7.2 19h9.6" />
        </svg>
      </span>
      <span>DSA Mentor</span>
      {children}
    </div>
  )
}

function TutorMessage({ msg, theme, currentTopicId, onFeedback, onOpenConfidence }) {
  const isAdaptive = Boolean(msg.meta?.is_adaptive || msg.action_taken === 'ADAPTIVE_CHECK')
  const shouldShow = msg.show_feedback || msg.feedback_rating != null || msg.feedback_label != null
  const rateText = msg.feedback_rating ? `Rated ${msg.feedback_rating}/10` : 'Rate confidence'
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(msg.content || '')
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch (_) {}
  }

  return (
    <div className={`message-row tutor-row${isAdaptive ? ' adaptive-intervention' : ''}`} data-message-id={msg.id}>
      <TutorLabel>
        {isAdaptive && <span className="adaptive-badge">Adaptive question</span>}
      </TutorLabel>
      <div className="message-text">
        <TutorContent content={msg.content} theme={theme} />
      </div>
      <div className="msg-actions">
        {msg.id && shouldShow && (
          <div className="feedback-bar" data-msg-id={msg.id}>
            <span className="feedback-label">Comprehension check</span>
            <button
              className={`feedback-btn${msg.feedback_label === 'clear' ? ' active' : ''}`}
              onClick={() => onFeedback({ messageId: msg.id, label: 'clear', conceptId: currentTopicId })}
            >👍 Clear</button>
            <button
              className={`feedback-btn${msg.feedback_label === 'confused' ? ' active' : ''}`}
              onClick={() => onFeedback({ messageId: msg.id, label: 'confused', conceptId: currentTopicId })}
            >🤔 Need help</button>
            <button
              className={`feedback-btn feedback-rate-btn${msg.feedback_rating ? ' active' : ''}`}
              onClick={() => onOpenConfidence(msg.id, currentTopicId)}
            >⭐ {rateText}</button>
          </div>
        )}
        <button className="msg-action-ghost" onClick={copy} title="Copy response">
          {copied
            ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5" /></svg>
            : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="12" height="12" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h10" /></svg>}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
    </div>
  )
}

function TutorContent({ content, theme }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '')
          return !inline && match ? (
            <SyntaxHighlighter
              style={theme === 'dark' ? oneDark : oneLight}
              language={match[1]}
              PreTag="div"
              customStyle={{ margin: 0, borderRadius: 12, fontSize: '0.84rem', background: 'var(--code-bg)', border: '1px solid var(--border)' }}
              codeTagProps={{ style: { fontFamily: 'var(--font-mono)', background: 'transparent' } }}
              {...props}
            >
              {String(children).replace(/\n$/, '')}
            </SyntaxHighlighter>
          ) : (
            <code className={className} {...props}>{children}</code>
          )
        }
      }}
    >
      {content || ''}
    </ReactMarkdown>
  )
}

function TopicTransitionCard({ msg, onFeedback, onOpenConfidence }) {
  const { old_topic_id, old_topic_name, new_topic_name } = msg.meta || {}
  const isResolved = msg.feedback_rating != null || msg.feedback_label != null

  const handleLabel = async (label) => {
    const rating = label === 'clear' ? 9 : 3
    await onFeedback({ messageId: msg.id, rating, label, conceptId: old_topic_id })
  }

  return (
    <div className={`topic-transition-card${isResolved ? ' resolved' : ''}`} data-msg-id={msg.id}>
      <div className="transition-card-header">
        <span className="transition-badge">Topic transition</span>
        <span className="transition-sub"><strong>{old_topic_name}</strong> → <strong>{new_topic_name}</strong></span>
      </div>
      <div className="transition-card-body">
        <p>{isResolved
          ? <>Recorded feedback for <strong>{old_topic_name}</strong>:</>
          : <>Before you move forward, how confident do you feel with <strong>{old_topic_name}</strong>?</>
        }</p>
        <div className="transition-actions">
          <button
            className={`btn-transition-feedback btn-rate-transition${msg.feedback_rating ? ' rated' : ''}`}
            disabled={isResolved}
            onClick={() => !isResolved && onOpenConfidence(msg.id, old_topic_id)}
          >{msg.feedback_rating ? `⭐ Rated ${msg.feedback_rating}/10` : `⭐ Rate ${old_topic_name} (1-10)`}</button>
          <button
            className={`btn-transition-feedback btn-label-transition${msg.feedback_label === 'confused' ? ' rated' : ''}`}
            data-label="confused"
            disabled={isResolved}
            onClick={() => !isResolved && handleLabel('confused')}
          >{msg.feedback_label === 'confused' ? '✓ Noted confusion' : '🤔 Still confused'}</button>
          <button
            className={`btn-transition-feedback btn-label-transition${msg.feedback_label === 'clear' ? ' rated' : ''}`}
            data-label="clear"
            disabled={isResolved}
            onClick={() => !isResolved && handleLabel('clear')}
          >{msg.feedback_label === 'clear' ? '✓ Marked mastered' : '👍 Mastered / clear'}</button>
        </div>
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="message-row tutor-row">
      <TutorLabel />
      <div className="typing-indicator" aria-label="Mentor is thinking">
        <span className="typing-text">Thinking</span>
        <span className="typing-dots"><i /><i /><i /></span>
      </div>
    </div>
  )
}

function SystemMessage({ content, isHtml }) {
  return (
    <div className="message-row system-row">
      <div className="system-pill">
        {isHtml ? <span dangerouslySetInnerHTML={{ __html: content }} /> : content}
      </div>
    </div>
  )
}
