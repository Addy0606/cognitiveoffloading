import React, { useState } from 'react'
import { CONFIDENCE_DESCRIPTIONS } from '../../utils.js'

export default function ConfidenceModal({ messageId, conceptId, onSave, onClose }) {
  const [rating, setRating] = useState(7)
  const [notes, setNotes] = useState('')

  const save = () => {
    onSave({ messageId, conceptId, rating, notes })
  }

  const conceptLabel = conceptId
    ? `Concept: ${conceptId.replace(/_/g, ' ').toUpperCase()}`
    : 'Rate Concept Confidence'

  return (
    <div className="modal-backdrop" id="confidenceModal" style={{ display: 'flex' }} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal-card">
        <div className="modal-header">
          <h3>Rate Your Concept Confidence</h3>
          <button className="modal-close" id="closeConfidenceModal" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <p id="modalConceptName">{conceptLabel}</p>
          <div className="rating-scale">
            <input
              type="range"
              id="confidenceSlider"
              min="1" max="10"
              value={rating}
              step="1"
              onChange={e => setRating(parseInt(e.target.value))}
            />
            <div className="rating-value-display">
              <span className="rating-big-num" id="confidenceNum">{rating}</span>
              <span className="rating-label-desc" id="confidenceLabel">{CONFIDENCE_DESCRIPTIONS[rating] || ''}</span>
            </div>
          </div>
          <div className="scale-labels">
            <span>1: Completely Lost</span>
            <span>5: Need Guidance</span>
            <span>10: Full Mastery</span>
          </div>
          <textarea
            id="confidenceNotes"
            placeholder="Optional notes on what clicked or what is still confusing..."
            rows="2"
            value={notes}
            onChange={e => setNotes(e.target.value)}
          />
        </div>
        <div className="modal-footer">
          <button className="btn btn-outline" id="cancelConfidenceBtn" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" id="saveConfidenceBtn" onClick={save}>Save Mastery State</button>
        </div>
      </div>
    </div>
  )
}

