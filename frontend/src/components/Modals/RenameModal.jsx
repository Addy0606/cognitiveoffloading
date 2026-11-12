import React, { useState } from 'react'

export default function RenameModal({ convId, currentTitle, onSave, onClose }) {
  const [title, setTitle] = useState(currentTitle || '')

  const save = () => {
    if (title.trim()) onSave(convId, title.trim())
  }

  return (
    <div className="modal-backdrop" id="renameModal" style={{ display: 'flex' }} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal-card">
        <div className="modal-header">
          <h3>Rename Conversation</h3>
          <button className="modal-close" id="closeRenameModal" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <input
            type="text"
            id="renameInput"
            className="text-input"
            placeholder="Enter conversation title..."
            value={title}
            onChange={e => setTitle(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') save(); if (e.key === 'Escape') onClose() }}
            autoFocus
          />
        </div>
        <div className="modal-footer">
          <button className="btn btn-outline" id="cancelRenameBtn" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" id="saveRenameBtn" onClick={save}>Save Title</button>
        </div>
      </div>
    </div>
  )
}

