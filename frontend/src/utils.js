export const CONFIDENCE_DESCRIPTIONS = {
  1: 'Completely Lost — Need full concept breakdown',
  2: 'Very Confused — Don\'t understand the core mechanism',
  3: 'Struggling — Understand premise, but can\'t solve it',
  4: 'Uncertain — Need step-by-step examples',
  5: 'Developing — Getting the idea with guidance',
  6: 'Fair — Can reason through with hints',
  7: 'Comfortable — Can explain basic approach and constraints',
  8: 'Confident — Can implement approach independently',
  9: 'Strong — Understand edge cases and time/space complexity',
  10: 'Mastered — Can teach this concept and optimize tradeoffs',
}

export function formatRelativeDate(isoStr) {
  if (!isoStr) return ''
  const date = new Date(isoStr)
  const now = new Date()
  const diffMins = Math.floor((now - date) / 60000)
  const diffHours = Math.floor(diffMins / 60)
  const diffDays = Math.floor(diffHours / 24)
  if (diffMins < 2) return 'Just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  if (diffDays === 1) return 'Yesterday'
  if (diffDays < 7) return `${diffDays}d ago`
  return date.toLocaleDateString()
}

export function formatTime(totalMins) {
  if (!totalMins || totalMins === 0) return '0m'
  if (totalMins >= 60) {
    const h = Math.floor(totalMins / 60)
    const m = Math.round(totalMins % 60)
    return `${h}h ${m}m`
  }
  return `${Math.round(totalMins)}m`
}

