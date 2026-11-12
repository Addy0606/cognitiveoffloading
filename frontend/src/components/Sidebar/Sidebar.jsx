import React from 'react'
import { formatRelativeDate } from '../../utils.js'

export default function Sidebar({
  conversations, currentConvId, mastery,
  onNewChat, onSelectConv, onDeleteConv, onRenameConv,
  onOpenTelemetry, onOpenAnalytics,
  isOpen, onToggleSidebar,
}) {
  return (
    <aside className={`sidebar${isOpen ? '' : ' collapsed'}`} id="sidebar" aria-hidden={!isOpen}>
      {/* Brand */}
      <div className="sidebar-header">
        <div className="brand">
          <span className="brand-icon" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="5" r="2.2" /><circle cx="5" cy="19" r="2.2" /><circle cx="19" cy="19" r="2.2" />
              <path d="M11 7 6 17M13 7l5 10M7.2 19h9.6" />
            </svg>
          </span>
          <span className="brand-name">CO DSA Mentor</span>
        </div>
        <button className="icon-btn sidebar-toggle" title="Collapse sidebar" aria-label="Collapse sidebar" onClick={onToggleSidebar}>
          <PanelIcon />
        </button>
      </div>

      <button id="newChatBtn" className="btn-new-chat" title="Start a fresh conversation" onClick={onNewChat}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>
        <span>New Chat</span>
      </button>

      {/* Nav */}
      <nav className="sidebar-nav">
        <button className="nav-item" onClick={onOpenAnalytics} title="View detailed topic breakdown & analytics">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></svg>
          <span>Concepts Explored</span>
          <span className="nav-badge" title="Mastered concepts">{mastery.masteredCount}</span>
        </button>
        <button id="openTelemetryBtn" className="nav-item" onClick={onOpenTelemetry} title="View server RL inference telemetry">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
          <span>Instructor Telemetry</span>
        </button>
      </nav>

      {/* Conversations */}
      <div className="sidebar-scrollable">
        <div className="section-title-row">
          <span className="section-heading">Recent Chats</span>
          <span className="section-badge" id="chatCountBadge">{conversations.length}</span>
        </div>
        <div className="conversations-list" id="conversationsList">
          {conversations.length === 0
            ? <div className="empty-state">No saved chats yet. Click "New Chat" to begin!</div>
            : conversations.map(c => (
              <ConvItem
                key={c.id}
                conv={c}
                isActive={c.id === currentConvId}
                onSelect={() => onSelectConv(c.id)}
                onRename={() => onRenameConv(c.id, c.title)}
                onDelete={() => onDeleteConv(c.id)}
              />
            ))
          }
        </div>
      </div>

      {/* Footer: mini mastery row (opens analytics) + theme toggle */}
      <div className="sidebar-footer">
        <div
          className="mastery-mini"
          id="masterySummaryCard"
          role="button"
          tabIndex="0"
          title={`${mastery.practiced} topics studied · Avg confidence ${mastery.avgConf}/10 — click for full breakdown`}
          onClick={onOpenAnalytics}
          onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpenAnalytics() } }}
        >
          <span className="mastery-avatar" aria-hidden="true">S</span>
          <div className="mastery-mini-body">
            <div className="mastery-mini-top">
              <span className="mastery-mini-name">Student</span>
              <span className="mastery-percent" id="masteryPercent">{mastery.masteredCount} mastered</span>
            </div>
            <div className="mastery-bar-bg">
              <div className="mastery-bar-fill" id="masteryBarFill" style={{ width: `${mastery.pct}%` }}></div>
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}

export function PanelIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="16" rx="2.5" /><path d="M9 4v16" />
    </svg>
  )
}

function ConvItem({ conv, isActive, onSelect, onRename, onDelete }) {
  const dateStr = formatRelativeDate(conv.updated_at)
  return (
    <div
      className={`conv-item${isActive ? ' active' : ''}`}
      data-conv-id={conv.id}
      onClick={onSelect}
      title={`${conv.title || 'DSA Chat'} — ${dateStr} • ${conv.message_count || 0} msgs`}
    >
      <span className="conv-title">{conv.title || 'DSA Chat'}</span>
      <div className="conv-actions">
        <button className="conv-action-btn btn-rename-conv" title="Rename conversation" aria-label="Rename conversation" onClick={e => { e.stopPropagation(); onRename() }}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" /></svg>
        </button>
        <button className="conv-action-btn btn-delete-conv" title="Delete conversation" aria-label="Delete conversation" onClick={e => { e.stopPropagation(); onDelete() }}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M18 6 6 18M6 6l12 12" /></svg>
        </button>
      </div>
    </div>
  )
}
