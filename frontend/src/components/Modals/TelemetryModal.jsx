import React from 'react'

const ACTION_NAMES = ['REDIRECT (0)', 'SCAFFOLD (1)', 'SOCRATIC (2)', 'DELAY (3)']
const SHORT_NAMES = ['REDIRECT', 'SCAFFOLD', 'SOCRATIC', 'DELAY']

export default function TelemetryModal({ data, onClose }) {
  const ext = parseFloat(data?.extraction_score || 0)
  const eng = parseFloat(data?.engagement_score || 0)
  const action = data?.action_taken || 'None'
  const actionId = data?.action_id ?? '--'
  const qVals = data?.q_values || [0, 0, 0, 0]
  const maxIdx = qVals.reduce((mi, v, i, arr) => v > arr[mi] ? i : mi, 0)
  // Raw policy choice (greedy = argmax Q). Older responses without the field fall back to argmax.
  const policyIdx = data ? (data.policy_action_id ?? maxIdx) : null
  const takenIdx = typeof data?.action_id === 'number' && data.action_id >= 0 ? data.action_id : null
  const overridden = Boolean(data?.guardrail_override) && policyIdx != null && takenIdx != null && policyIdx !== takenIdx
  const usedPolicy = takenIdx != null && qVals.some(v => v !== 0)

  let actionDesc = 'Action determined by PyTorch Q-Network'
  if (data) {
    if (overridden) actionDesc = `Policy chose ${SHORT_NAMES[policyIdx]}; overridden by rule-based guardrail`
    else if (usedPolicy) actionDesc = `Action ID ${actionId} selected by PyTorch Q-Network`
    else actionDesc = `Action ID ${actionId} (conversational turn, policy not used)`
  }

  return (
    <div className="modal-backdrop" id="telemetryModal" style={{ display: 'flex' }} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal-card modal-large">
        <div className="modal-header">
          <div>
            <h3>Server RL Policy &amp; DeBERTa Telemetry</h3>
            <p className="modal-sub">Live telemetry logged to server console per teacher specification</p>
          </div>
          <button className="modal-close" id="closeTelemetryModal" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <div className="telemetry-grid">
            <div className="t-card">
              <span className="t-label">Extraction Probability (Offloading)</span>
              <div className="t-val" id="tExtractionVal">{data ? ext.toFixed(3) : '--'}</div>
              <div className="t-bar-bg"><div className="t-bar-fill fill-red" id="tExtractionBar" style={{ width: `${Math.min(100, ext * 100)}%` }}></div></div>
              <span className="t-sub">DeBERTa-v3 detection score (0 to 1)</span>
            </div>
            <div className="t-card">
              <span className="t-label">Engagement &amp; Reasoning Effort</span>
              <div className="t-val" id="tEngagementVal">{data ? eng.toFixed(3) : '--'}</div>
              <div className="t-bar-bg"><div className="t-bar-fill fill-green" id="tEngagementBar" style={{ width: `${Math.min(100, eng * 100)}%` }}></div></div>
              <span className="t-sub">Calculated student effort &amp; questions signal</span>
            </div>
            <div className="t-card">
              <span className="t-label">Last RL Intervention Action</span>
              <div className="t-val-action" id="tActionVal">{action}</div>
              {overridden && <span className="override-badge" id="tOverrideBadge">Guardrail override</span>}
              <span className="t-sub" id="tActionDesc">{actionDesc}</span>
            </div>
          </div>

          <div className="q-values-table">
            <h4>Q-Values Breakdown (Action Selection Network)</h4>
            <div className="q-row-grid">
              {ACTION_NAMES.map((name, i) => {
                const isPolicy = data && usedPolicy && i === policyIdx
                const isTaken = data && i === takenIdx
                return (
                  <div
                    key={i}
                    className={`q-box${isPolicy ? ' best' : ''}${isTaken && overridden ? ' taken' : ''}`}
                    id={`qBox${i}`}
                  >
                    <span className="q-name">{name}</span>
                    <span className="q-num" id={`qNum${i}`}>{(qVals[i] || 0).toFixed(3)}</span>
                    {(isPolicy || (isTaken && overridden)) && (
                      <span className="q-tags">
                        {isPolicy && <span className="q-tag q-tag-policy">Policy pick</span>}
                        {isTaken && overridden && <span className="q-tag q-tag-taken">Taken</span>}
                      </span>
                    )}
                  </div>
                )
              })}
            </div>
            {overridden && (
              <p className="q-override-note">
                The Q-network preferred <strong>{SHORT_NAMES[policyIdx]}</strong>, but a safety guardrail
                replaced it with <strong>{SHORT_NAMES[takenIdx]}</strong> (e.g. extraction &gt; 0.7 always redirects).
              </p>
            )}
          </div>

          <div className="terminal-log-hint">
            <code>💡 Tip: Detailed telemetry with student turn states is continuously printed to the server terminal.</code>
          </div>
        </div>
      </div>
    </div>
  )
}
