// API layer — all calls to FastAPI backend

const BASE = ''  // proxied to :5000 by Vite dev server; empty for production

export async function fetchConversations() {
  const r = await fetch(`${BASE}/api/conversations`)
  return r.json()
}

export async function fetchConversationMessages(convId) {
  const r = await fetch(`${BASE}/api/conversations/${convId}`)
  if (!r.ok) throw new Error('Conversation not found')
  return r.json()
}

export async function deleteConversation(convId) {
  await fetch(`${BASE}/api/conversations/${convId}`, { method: 'DELETE' })
}

export async function renameConversation(convId, title) {
  const r = await fetch(`${BASE}/api/conversations/${convId}/title`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title })
  })
  return r.json()
}

export async function sendChatMessage({ message, conversation_id, topic_id, active_dwell_seconds }) {
  const r = await fetch(`${BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, conversation_id, topic_id, use_policy: true, active_dwell_seconds })
  })
  return r.json()
}

export async function sendFeedback({ message_id, rating, label, concept_id, conversation_id }) {
  const r = await fetch(`${BASE}/api/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message_id, rating, label, concept_id, conversation_id })
  })
  return r.json()
}

export async function sendHeartbeat({ topic_id, conversation_id, active_seconds }) {
  try {
    await fetch(`${BASE}/api/heartbeat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic_id, conversation_id, active_seconds })
    })
  } catch (_) { /* silent */ }
}

export async function fetchConcepts() {
  const r = await fetch(`${BASE}/api/concepts`)
  return r.json()
}

export async function fetchConceptAnalytics() {
  const r = await fetch(`${BASE}/api/concepts/analytics`)
  if (!r.ok) throw new Error('analytics not available')
  return r.json()
}

export async function fetchCheckin() {
  const r = await fetch(`${BASE}/api/checkin`)
  return r.json()
}

export async function resolveCheckin({ concept_id, confidence_score, notes }) {
  await fetch(`${BASE}/api/checkin/resolve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ concept_id, confidence_score, notes })
  })
}

export async function fetchStatus() {
  const r = await fetch(`${BASE}/api/status`)
  return r.json()
}

