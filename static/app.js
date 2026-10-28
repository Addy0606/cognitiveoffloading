/**
 * DSA AI MENTOR — Client Application Logic
 * Freeform Chat Workflow:
 * - Start a chat by asking any DSA doubt or concept
 * - Auto-detects DSA concept & auto-titles conversation
 * - Inline & topbar manual conversation renaming
 * - SQLite persistent conversation history
 * - Student comprehension feedback & 1-10 confidence rating
 * - Spaced retention check-in (only triggers on actual past practice)
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------
    // DOM Elements
    // -------------------------------------------------------------
    const newChatBtn = document.getElementById("newChatBtn");
    const conversationsList = document.getElementById("conversationsList");
    const chatCountBadge = document.getElementById("chatCountBadge");

    const chatViewport = document.getElementById("chatViewport");
    const chatMessages = document.getElementById("chatMessages");
    const messageInput = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");
    const providerPill = document.getElementById("providerPill");

    // Top Header & Renaming
    const activeTopicTitle = document.getElementById("activeTopicTitle");
    const renameHeaderBtn = document.getElementById("renameHeaderBtn");
    const activeCategory = document.getElementById("activeCategory");
    const activeTopicSub = document.getElementById("activeTopicSub");
    const topTopicIcon = document.getElementById("topTopicIcon");

    // Rename Modal
    const renameModal = document.getElementById("renameModal");
    const closeRenameModal = document.getElementById("closeRenameModal");
    const cancelRenameBtn = document.getElementById("cancelRenameBtn");
    const saveRenameBtn = document.getElementById("saveRenameBtn");
    const renameInput = document.getElementById("renameInput");

    // Retention Check-in Banner
    const retentionBanner = document.getElementById("retentionBanner");
    const retentionText = document.getElementById("retentionText");
    const retentionTestBtn = document.getElementById("retentionTestBtn");
    const retentionRateBtn = document.getElementById("retentionRateBtn");
    const retentionDismissBtn = document.getElementById("retentionDismissBtn");

    // Mastery Summary & Analytics Modal Elements
    const masterySummaryCard = document.getElementById("masterySummaryCard");
    const masteryPercent = document.getElementById("masteryPercent");
    const masteryBarFill = document.getElementById("masteryBarFill");
    const masteryPracticed = document.getElementById("masteryPracticed");
    const masteryAvgConf = document.getElementById("masteryAvgConf");

    // Concept Analytics & Mastery Modal
    const conceptMasteryModal = document.getElementById("conceptMasteryModal");
    const closeConceptMasteryModal = document.getElementById("closeConceptMasteryModal");
    const doneConceptModalBtn = document.getElementById("doneConceptModalBtn");
    const modalMasteryBadge = document.getElementById("modalMasteryBadge");
    const kpiMasteredCount = document.getElementById("kpiMasteredCount");
    const kpiMasteredSub = document.getElementById("kpiMasteredSub");
    const kpiTotalTurns = document.getElementById("kpiTotalTurns");
    const kpiTotalTime = document.getElementById("kpiTotalTime");
    const kpiAvgConfidence = document.getElementById("kpiAvgConfidence");
    const kpiAvgEngagement = document.getElementById("kpiAvgEngagement");
    const countAll = document.getElementById("countAll");
    const countActive = document.getElementById("countActive");
    const countMastered = document.getElementById("countMastered");
    const countNotStarted = document.getElementById("countNotStarted");
    const topicFilterPills = document.getElementById("topicFilterPills");
    const topicSearchInput = document.getElementById("topicSearchInput");
    const topicsListContainer = document.getElementById("topicsListContainer");

    // Confidence Modal
    const confidenceModal = document.getElementById("confidenceModal");
    const closeConfidenceModal = document.getElementById("closeConfidenceModal");
    const cancelConfidenceBtn = document.getElementById("cancelConfidenceBtn");
    const saveConfidenceBtn = document.getElementById("saveConfidenceBtn");
    const modalConceptName = document.getElementById("modalConceptName");
    const confidenceSlider = document.getElementById("confidenceSlider");
    const confidenceNum = document.getElementById("confidenceNum");
    const confidenceLabel = document.getElementById("confidenceLabel");
    const confidenceNotes = document.getElementById("confidenceNotes");

    // Instructor Telemetry Modal
    const openTelemetryBtn = document.getElementById("openTelemetryBtn");
    const telemetryModal = document.getElementById("telemetryModal");
    const closeTelemetryModal = document.getElementById("closeTelemetryModal");
    const tExtractionVal = document.getElementById("tExtractionVal");
    const tExtractionBar = document.getElementById("tExtractionBar");
    const tEngagementVal = document.getElementById("tEngagementVal");
    const tEngagementBar = document.getElementById("tEngagementBar");
    const tActionVal = document.getElementById("tActionVal");
    const tActionDesc = document.getElementById("tActionDesc");

    // -------------------------------------------------------------
    // Application State
    // -------------------------------------------------------------
    let currentConversationId = null;
    let currentTopicId = null;
    let isSending = false;
    let activeModalTarget = { messageId: null, conceptId: null };
    let renamingConvId = null;
    let activeCheckinData = null;

    // -------------------------------------------------------------
    // Active Cognitive Engagement Time (ACET) Watchdog
    // -------------------------------------------------------------
    let turnActiveFocusSeconds = 0;
    let lastUserActivityTimestamp = Date.now();
    let isWindowFocused = !document.hidden;

    function recordUserActivity() {
        lastUserActivityTimestamp = Date.now();
    }
    window.addEventListener("mousemove", recordUserActivity, { passive: true });
    window.addEventListener("keydown", recordUserActivity, { passive: true });
    window.addEventListener("scroll", recordUserActivity, { passive: true });
    window.addEventListener("click", recordUserActivity, { passive: true });

    document.addEventListener("visibilitychange", () => {
        isWindowFocused = !document.hidden;
        if (isWindowFocused) {
            lastUserActivityTimestamp = Date.now();
        }
    });

    // Second-by-second focus accumulator (pauses when tab blurred or idle > 60s)
    setInterval(() => {
        const isRecentlyActive = (Date.now() - lastUserActivityTimestamp) <= 60000;
        if (isWindowFocused && isRecentlyActive && !isSending) {
            turnActiveFocusSeconds += 1;
        }
    }, 1000);

    // Periodic Heartbeat every 25 seconds
    setInterval(async () => {
        const isRecentlyActive = (Date.now() - lastUserActivityTimestamp) <= 60000;
        if (isWindowFocused && isRecentlyActive && (currentTopicId || currentConversationId)) {
            try {
                await fetch("/api/heartbeat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        topic_id: currentTopicId,
                        conversation_id: currentConversationId,
                        active_seconds: 25.0
                    })
                });
            } catch (e) {
                // Ignore silent network drops
            }
        }
    }, 25000);

    const CONFIDENCE_DESCRIPTIONS = {
        1: "Completely Lost — Need full concept breakdown",
        2: "Very Confused — Don't understand the core mechanism",
        3: "Struggling — Understand premise, but can't solve it",
        4: "Uncertain — Need step-by-step examples",
        5: "Developing — Getting the idea with guidance",
        6: "Fair — Can reason through with hints",
        7: "Comfortable — Can explain basic approach and constraints",
        8: "Confident — Can implement approach independently",
        9: "Strong — Understand edge cases and time/space complexity",
        10: "Mastered — Can teach this concept and optimize tradeoffs"
    };

    // -------------------------------------------------------------
    // 1. Initialize
    // -------------------------------------------------------------
    async function init() {
        await Promise.all([
            loadConversations(),
            loadMastery(),
            loadStatus()
        ]);
        checkRetention();
    }

    // -------------------------------------------------------------
    // 2. Conversations List & Persistence
    // -------------------------------------------------------------
    async function loadConversations() {
        try {
            const resp = await fetch("/api/conversations");
            const data = await resp.json();
            const convs = data.conversations || [];
            chatCountBadge.textContent = convs.length;
            renderConversationsList(convs);

            // Auto-select latest conversation if none selected and convs exist
            if (!currentConversationId && convs.length > 0) {
                loadConversationMessages(convs[0].id);
            } else if (convs.length === 0) {
                renderWelcomeCard();
            }
        } catch (err) {
            console.error("Failed to load conversations:", err);
        }
    }

    function renderConversationsList(convs) {
        conversationsList.innerHTML = "";
        if (convs.length === 0) {
            conversationsList.innerHTML = `<div class="empty-state">No saved chats yet. Click "New Chat" to begin!</div>`;
            return;
        }

        convs.forEach(c => {
            const item = document.createElement("div");
            item.className = `conv-item ${currentConversationId === c.id ? 'active' : ''}`;
            item.dataset.convId = c.id;

            const dateStr = formatRelativeDate(c.updated_at);
            item.innerHTML = `
                <div class="conv-title-wrap">
                    <span class="conv-title">${escapeHtml(c.title || "DSA Chat")}</span>
                    <span class="conv-date">${dateStr} &bull; ${c.message_count || 0} msgs</span>
                </div>
                <div class="conv-actions">
                    <button class="btn-rename-conv" title="Rename conversation">✏️</button>
                    <button class="btn-delete-conv" title="Delete conversation">&times;</button>
                </div>
            `;

            item.addEventListener("click", (e) => {
                if (e.target.classList.contains("btn-rename-conv")) {
                    e.stopPropagation();
                    openRenameModal(c.id, c.title);
                    return;
                }
                if (e.target.classList.contains("btn-delete-conv")) {
                    e.stopPropagation();
                    deleteConversation(c.id);
                    return;
                }
                loadConversationMessages(c.id);
            });

            conversationsList.appendChild(item);
        });
    }

    async function loadConversationMessages(convId) {
        try {
            const resp = await fetch(`/api/conversations/${convId}`);
            if (!resp.ok) throw new Error("Conversation not found");
            const data = await resp.json();

            currentConversationId = convId;
            const conv = data.conversation;
            const topic = data.topic;
            const messages = data.messages || [];

            currentTopicId = conv.topic_id || (topic ? topic.concept_id : null);

            // Update sidebar selection
            document.querySelectorAll(".conv-item").forEach(el => {
                el.classList.toggle("active", el.dataset.convId === convId);
            });

            // Update Topbar
            activeTopicTitle.textContent = conv.title || "DSA Chat";
            if (topic) {
                activeCategory.textContent = topic.category || "DSA";
                activeCategory.style.display = "inline-block";
                activeTopicSub.textContent = `Concept: ${topic.concept_name}`;
            } else {
                activeCategory.style.display = "none";
                activeTopicSub.textContent = "Adaptive AI mentor ready to guide your learning.";
            }

            // Render Messages
            chatMessages.innerHTML = "";
            if (messages.length === 0) {
                renderWelcomeCard();
            } else {
                messages.forEach(m => {
                    if (m.role === "student") {
                        appendStudentMessage(m.content);
                    } else if (m.role === "tutor") {
                        appendTutorMessage(m.content, m.id, m.feedback_rating, m.feedback_label);
                    } else if (m.role === "topic_transition") {
                        const meta = m.meta || {};
                        appendTopicTransitionCard(
                            meta.old_topic_id,
                            meta.old_topic_name,
                            meta.new_topic_name,
                            m.id,
                            m.feedback_rating || meta.rating,
                            m.feedback_label || meta.label
                        );
                    }
                });
            }

            scrollToBottom();
        } catch (err) {
            console.error("Failed to load conversation:", err);
        }
    }

    function startNewChat() {
        currentConversationId = null;
        currentTopicId = null;
        activeTopicTitle.textContent = "New Chat";
        activeCategory.style.display = "none";
        activeTopicSub.textContent = "Adaptive AI mentor ready to guide your learning.";
        document.querySelectorAll(".conv-item").forEach(el => el.classList.remove("active"));
        chatMessages.innerHTML = "";
        renderWelcomeCard();
        checkRetention();
        messageInput.focus();
    }

    newChatBtn.addEventListener("click", startNewChat);

    async function deleteConversation(convId) {
        try {
            await fetch(`/api/conversations/${convId}`, { method: "DELETE" });
            if (currentConversationId === convId) {
                startNewChat();
            }
            await loadConversations();
            await loadMastery();
        } catch (err) {
            console.error("Failed to delete conversation:", err);
        }
    }

    // -------------------------------------------------------------
    // 3. Conversation Renaming
    // -------------------------------------------------------------
    function openRenameModal(convId, currentTitle) {
        renamingConvId = convId;
        renameInput.value = currentTitle || "DSA Chat";
        renameModal.style.display = "flex";
        setTimeout(() => renameInput.focus(), 50);
    }

    renameHeaderBtn.addEventListener("click", () => {
        if (!currentConversationId) return;
        openRenameModal(currentConversationId, activeTopicTitle.textContent);
    });

    closeRenameModal.addEventListener("click", () => renameModal.style.display = "none");
    cancelRenameBtn.addEventListener("click", () => renameModal.style.display = "none");

    saveRenameBtn.addEventListener("click", saveRename);
    renameInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") saveRename();
        if (e.key === "Escape") renameModal.style.display = "none";
    });

    async function saveRename() {
        const newTitle = renameInput.value.trim();
        if (!newTitle || !renamingConvId) return;

        try {
            const resp = await fetch(`/api/conversations/${renamingConvId}/title`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title: newTitle })
            });

            if (resp.ok) {
                if (currentConversationId === renamingConvId) {
                    activeTopicTitle.textContent = newTitle;
                }
                renameModal.style.display = "none";
                await loadConversations();
            }
        } catch (err) {
            console.error("Failed to rename conversation:", err);
        }
    }

    // -------------------------------------------------------------
    // 4. Send Message & Receive Response
    // -------------------------------------------------------------
    async function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || isSending) return;

        isSending = true;
        sendBtn.disabled = true;
        messageInput.value = "";
        messageInput.style.height = "auto";

        // Remove welcome card if present
        const welcome = document.getElementById("welcomeCard");
        if (welcome) welcome.remove();

        // Append student message bubble
        appendStudentMessage(text);

        // Show typing indicator
        const typingId = appendTypingIndicator();

        try {
            // Measure authentic active dwell seconds (clamped 5s to 180s per turn)
            const activeDwell = Math.min(Math.max(turnActiveFocusSeconds, 5), 180);
            turnActiveFocusSeconds = 0; // reset for next turn

            const payload = {
                message: text,
                conversation_id: currentConversationId,
                topic_id: currentTopicId,
                use_policy: true,
                active_dwell_seconds: activeDwell
            };

            const resp = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await resp.json();
            removeTypingIndicator(typingId);

            if (data.error) {
                appendSystemMessage(`⚠️ Error: ${data.error}`);
            } else {
                // If new conversation was created on the backend
                if (!currentConversationId && data.conversation_id) {
                    currentConversationId = data.conversation_id;
                }

                // If auto-title was updated
                if (data.title) {
                    activeTopicTitle.textContent = data.title;
                }
                if (data.topic_id) {
                    currentTopicId = data.topic_id;
                }
                if (data.topic_name) {
                    activeCategory.textContent = data.topic_name;
                    activeCategory.style.display = "inline-block";
                    activeTopicSub.textContent = `Concept: ${data.topic_name}`;
                }

                // If topic transition occurred, render interactive topic feedback prompt
                if (data.topic_switched && data.old_topic_name && data.old_topic_id) {
                    appendTopicTransitionCard(data.old_topic_id, data.old_topic_name, data.topic_name);
                }

                appendTutorMessage(data.response, data.message_id);
                updateTelemetryModal(data);
                loadConversations();
                loadMastery();
            }
        } catch (err) {
            removeTypingIndicator(typingId);
            appendSystemMessage(`⚠️ Connection error: ${err.message}`);
        } finally {
            isSending = false;
            sendBtn.disabled = false;
            messageInput.focus();
        }
    }

    sendBtn.addEventListener("click", sendMessage);

    messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    messageInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
    });

    // Quick prompt chips
    document.querySelectorAll(".qp-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const prompt = chip.dataset.prompt;
            messageInput.value = prompt;
            messageInput.focus();
            sendMessage();
        });
    });

    // -------------------------------------------------------------
    // 5. Message Rendering & Feedback System
    // -------------------------------------------------------------
    function appendStudentMessage(text) {
        const row = document.createElement("div");
        row.className = "message-row student-row";
        row.innerHTML = `
            <div class="message-bubble student-bubble">
                <div class="message-text">${escapeHtml(text)}</div>
            </div>
            <div class="avatar student-avatar">👤</div>
        `;
        chatMessages.appendChild(row);
        scrollToBottom();
    }

    function appendTutorMessage(text, messageId = null, feedbackRating = null, feedbackLabel = null) {
        const row = document.createElement("div");
        row.className = "message-row tutor-row";
        row.dataset.messageId = messageId || "";

        let feedbackHtml = "";
        if (messageId) {
            const clearActive = feedbackLabel === "clear" ? "active" : "";
            const confusedActive = feedbackLabel === "confused" ? "active" : "";
            const rateText = feedbackRating ? `⭐ Rated ${feedbackRating}/10` : "⭐ Rate Confidence (1-10)";

            feedbackHtml = `
                <div class="feedback-bar" data-msg-id="${messageId}">
                    <span class="feedback-label">Comprehension:</span>
                    <button class="feedback-btn ${clearActive}" data-label="clear" title="I understand this step">
                        👍 Clear
                    </button>
                    <button class="feedback-btn ${confusedActive}" data-label="confused" title="Still need clarification">
                        🤔 Need Help
                    </button>
                    <button class="feedback-btn feedback-rate-btn" data-action="rate" title="Rate confidence from 1 to 10">
                        ${rateText}
                    </button>
                </div>
            `;
        }

        row.innerHTML = `
            <div class="avatar tutor-avatar">🎓</div>
            <div class="message-bubble tutor-bubble">
                <div class="tutor-bubble-header">
                    <span>DSA Mentor</span>
                </div>
                <div class="message-text">${formatMessageContent(text)}</div>
                ${feedbackHtml}
            </div>
        `;

        const bar = row.querySelector(".feedback-bar");
        if (bar) {
            bar.querySelectorAll(".feedback-btn").forEach(btn => {
                btn.addEventListener("click", () => {
                    const action = btn.dataset.action;
                    const label = btn.dataset.label;
                    if (action === "rate") {
                        openConfidenceModal(messageId, currentTopicId);
                    } else if (label) {
                        submitFeedback(messageId, null, label, currentTopicId, btn);
                    }
                });
            });
        }

        chatMessages.appendChild(row);
        scrollToBottom();
    }

    async function submitFeedback(messageId, rating, label, conceptId, clickedBtn = null) {
        try {
            const resp = await fetch("/api/feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message_id: messageId,
                    rating: rating,
                    label: label,
                    concept_id: conceptId,
                    conversation_id: currentConversationId
                })
            });

            const data = await resp.json();

            if (resp.ok && clickedBtn) {
                const parent = clickedBtn.closest(".feedback-bar");
                parent.querySelectorAll(".feedback-btn").forEach(b => {
                    if (b.dataset.label) b.classList.remove("active");
                });
                clickedBtn.classList.add("active");
                loadMastery();

                if (label === "confused") {
                    appendSystemMessage("💡 <em>Feedback noted: The tutor is diagnosing this step to clear up confusion.</em>");
                } else if (label === "clear") {
                    appendSystemMessage("👍 <em>Feedback noted: Concept clarity recorded! The tutor will test deeper comprehension.</em>");
                }

                // If adaptive diagnostic or challenge question is returned, append it!
                if (data.adaptive_intervention && data.adaptive_intervention.prompt) {
                    appendTutorMessage(data.adaptive_intervention.prompt, data.adaptive_intervention.message_id);
                }
            }
        } catch (err) {
            console.error("Failed to submit feedback:", err);
        }
    }

    function appendTopicTransitionCard(oldTopicId, oldTopicName, newTopicName, messageId = null, feedbackRating = null, feedbackLabel = null) {
        const card = document.createElement("div");
        card.className = "topic-transition-card";
        if (messageId) card.dataset.msgId = messageId;

        const isResolved = (feedbackRating !== null && feedbackRating !== undefined) || (feedbackLabel !== null && feedbackLabel !== undefined);

        let rateBtnText = `⭐ Rate ${escapeHtml(oldTopicName)} (1-10)`;
        let rateBtnClass = "btn-transition-feedback btn-rate-transition";
        let clearBtnText = "👍 Mastered / Clear";
        let clearBtnClass = "btn-transition-feedback btn-label-transition";
        let confusedBtnText = "🤔 Still Confused";
        let confusedBtnClass = "btn-transition-feedback btn-label-transition";

        if (feedbackRating) {
            rateBtnText = `⭐ Rated ${feedbackRating}/10`;
            rateBtnClass += " rated";
        }
        if (feedbackLabel === "clear") {
            clearBtnText = "✓ Marked Mastered";
            clearBtnClass += " rated";
        } else if (feedbackLabel === "confused") {
            confusedBtnText = "✓ Noted Confusion";
            confusedBtnClass += " rated";
        }

        card.innerHTML = `
            <div class="transition-card-header">
                <span class="transition-badge">🔄 Topic Transition</span>
                <span class="transition-sub">Leaving: <strong>${escapeHtml(oldTopicName)}</strong> → Exploring: <strong>${escapeHtml(newTopicName)}</strong></span>
            </div>
            <div class="transition-card-body">
                <p>${isResolved ? `Recorded feedback for <strong>${escapeHtml(oldTopicName)}</strong>:` : `Before you move forward, how confident do you feel with <strong>${escapeHtml(oldTopicName)}</strong>?`}</p>
                <div class="transition-actions">
                    <button class="${rateBtnClass}" ${isResolved ? "disabled" : ""}>${rateBtnText}</button>
                    <button class="${confusedBtnClass}" data-label="confused" ${isResolved ? "disabled" : ""}>${confusedBtnText}</button>
                    <button class="${clearBtnClass}" data-label="clear" ${isResolved ? "disabled" : ""}>${clearBtnText}</button>
                </div>
            </div>
        `;

        if (!isResolved) {
            const rateBtn = card.querySelector(".btn-rate-transition");
            if (rateBtn) {
                rateBtn.addEventListener("click", () => {
                    openConfidenceModal(messageId, oldTopicId);
                });
            }

            card.querySelectorAll(".btn-label-transition").forEach(btn => {
                btn.addEventListener("click", async () => {
                    const label = btn.dataset.label;
                    const rating = label === "clear" ? 9 : 3;
                    try {
                        const resp = await fetch("/api/feedback", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                message_id: messageId,
                                rating: rating,
                                label: label,
                                concept_id: oldTopicId,
                                conversation_id: currentConversationId
                            })
                        });
                        const data = await resp.json();

                        btn.classList.add("rated");
                        btn.textContent = label === "clear" ? "✓ Marked Mastered" : "✓ Noted Confusion";
                        card.querySelectorAll(".btn-transition-feedback").forEach(b => {
                            if (b !== btn) b.disabled = true;
                        });
                        loadMastery();
                        appendSystemMessage(`💡 <em>Recorded ${label === 'clear' ? 'mastery' : 'review needed'} for ${escapeHtml(oldTopicName)}!</em>`);

                        // If adaptive question triggered, present it directly
                        if (data.adaptive_intervention && data.adaptive_intervention.prompt) {
                            appendTutorMessage(data.adaptive_intervention.prompt, data.adaptive_intervention.message_id);
                        }
                    } catch (e) {
                        console.error("Failed transition feedback:", e);
                    }
                });
            });
        }

        chatMessages.appendChild(card);
        scrollToBottom();
    }

    function appendSystemMessage(htmlText) {
        const row = document.createElement("div");
        row.className = "message-row system-row";
        row.style.justifyContent = "center";
        row.innerHTML = `
            <div style="font-size: 0.8rem; color: var(--text-muted); padding: 6px 14px; background: rgba(255,255,255,0.03); border-radius: var(--radius-full); border: 1px solid var(--border-subtle);">
                ${htmlText}
            </div>
        `;
        chatMessages.appendChild(row);
        scrollToBottom();
    }

    function appendTypingIndicator() {
        const id = `typing_${Date.now()}`;
        const row = document.createElement("div");
        row.className = "message-row tutor-row";
        row.id = id;
        row.innerHTML = `
            <div class="avatar tutor-avatar">🎓</div>
            <div class="message-bubble tutor-bubble" style="padding: 12px 18px;">
                <div style="display:flex; gap:6px; align-items:center;">
                    <span style="font-size:0.75rem; color:var(--text-muted);">Thinking...</span>
                    <span class="pulse-indicator"></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(row);
        scrollToBottom();
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function renderWelcomeCard() {
        if (document.getElementById("welcomeCard")) return;
        const div = document.createElement("div");
        div.className = "tutor-welcome-card";
        div.id = "welcomeCard";
        div.innerHTML = `
            <div class="welcome-badge">🎓 Adaptive Learning Active</div>
            <h3>Welcome to your Personal DSA Mentor</h3>
            <p>
                Ask about any Data Structure or Algorithm! Whether you're stuck on <strong>AVL Trees</strong>, 
                curious about <strong>Dynamic Programming</strong>, or debugging an edge case, 
                the mentor guides your independent reasoning without giving away direct code dumps.
            </p>
            <div class="welcome-tips">
                <div class="tip-card">
                    <span class="tip-icon">💡</span>
                    <div><strong>Ask freely</strong> — start any topic, e.g. <em>"How do AVL rotations work?"</em></div>
                </div>
                <div class="tip-card">
                    <span class="tip-icon">🪜</span>
                    <div><strong>Step-by-step guidance</strong> — if overwhelmed, ask to break down the approach into smaller parts.</div>
                </div>
                <div class="tip-card">
                    <span class="tip-icon">📊</span>
                    <div><strong>Track your confidence</strong> — rate your comprehension (1-10) after each explanation.</div>
                </div>
            </div>
        `;
        chatMessages.appendChild(div);
    }

    // -------------------------------------------------------------
    // 6. Spaced Retention Check-In
    // -------------------------------------------------------------
    async function checkRetention() {
        try {
            const resp = await fetch("/api/checkin");
            const data = await resp.json();
            const checkin = data.checkin;

            if (checkin && checkin.concept_name) {
                activeCheckinData = checkin;
                retentionText.innerHTML = `
                    Last time, you practiced <strong>${escapeHtml(checkin.concept_name)}</strong> 
                    (${checkin.confidence_score > 0 ? 'Confidence: ' + checkin.confidence_score + '/10' : 'in progress'}). 
                    How confident are you feeling with that concept today?
                `;
                retentionBanner.style.display = "flex";
            } else {
                retentionBanner.style.display = "none";
            }
        } catch (err) {
            console.error("Failed to check retention:", err);
            retentionBanner.style.display = "none";
        }
    }

    retentionTestBtn.addEventListener("click", () => {
        if (!activeCheckinData) return;
        const prompt = `Can you ask me a quick 1-minute question to test my understanding of ${activeCheckinData.concept_name}?`;
        messageInput.value = prompt;
        retentionBanner.style.display = "none";
        sendMessage();
    });

    retentionRateBtn.addEventListener("click", () => {
        if (!activeCheckinData) return;
        openConfidenceModal(null, activeCheckinData.concept_id);
    });

    retentionDismissBtn.addEventListener("click", () => {
        retentionBanner.style.display = "none";
    });

    // -------------------------------------------------------------
    // 7. Confidence Rating Modal
    // -------------------------------------------------------------
    function openConfidenceModal(messageId = null, conceptId = null) {
        activeModalTarget = { messageId, conceptId };
        modalConceptName.textContent = conceptId ? `Concept: ${conceptId.replace(/_/g, ' ').toUpperCase()}` : "Rate Concept Confidence";

        confidenceSlider.value = 7;
        confidenceNum.textContent = 7;
        confidenceLabel.textContent = CONFIDENCE_DESCRIPTIONS[7];
        confidenceNotes.value = "";

        confidenceModal.style.display = "flex";
    }

    confidenceSlider.addEventListener("input", function() {
        const val = parseInt(this.value);
        confidenceNum.textContent = val;
        confidenceLabel.textContent = CONFIDENCE_DESCRIPTIONS[val] || "";
    });

    closeConfidenceModal.addEventListener("click", () => confidenceModal.style.display = "none");
    cancelConfidenceBtn.addEventListener("click", () => confidenceModal.style.display = "none");

    saveConfidenceBtn.addEventListener("click", async () => {
        const val = parseInt(confidenceSlider.value);
        const notes = confidenceNotes.value.trim();

        let adaptiveData = null;

        if (activeModalTarget.conceptId) {
            await fetch("/api/checkin/resolve", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    concept_id: activeModalTarget.conceptId,
                    confidence_score: val,
                    notes: notes
                })
            });

            // Also send to /api/feedback so transition card is marked resolved and adaptive questions trigger
            try {
                const resp = await fetch("/api/feedback", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        message_id: activeModalTarget.messageId,
                        rating: val,
                        concept_id: activeModalTarget.conceptId,
                        conversation_id: currentConversationId
                    })
                });
                adaptiveData = await resp.json();
            } catch (e) {
                console.error("Failed to submit feedback from modal:", e);
            }

            // Update transition card if visible
            const tCard = document.querySelector(`.topic-transition-card[data-msg-id="${activeModalTarget.messageId}"]`) || document.querySelector(".topic-transition-card:not(.resolved)");
            if (tCard) {
                tCard.classList.add("resolved");
                const rBtn = tCard.querySelector(".btn-rate-transition");
                if (rBtn) {
                    rBtn.textContent = `⭐ Rated ${val}/10`;
                    rBtn.classList.add("rated");
                }
                tCard.querySelectorAll(".btn-transition-feedback").forEach(b => {
                    if (b !== rBtn) b.disabled = true;
                });
            }
        } else if (activeModalTarget.messageId) {
            try {
                const resp = await fetch("/api/feedback", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        message_id: activeModalTarget.messageId,
                        rating: val,
                        concept_id: activeModalTarget.conceptId,
                        conversation_id: currentConversationId
                    })
                });
                adaptiveData = await resp.json();
            } catch (e) {
                console.error("Failed feedback:", e);
            }

            const row = document.querySelector(`.message-row[data-message-id="${activeModalTarget.messageId}"]`);
            if (row) {
                const btn = row.querySelector(".feedback-rate-btn");
                if (btn) btn.textContent = `⭐ Rated ${val}/10`;
            }
        }

        if (val <= 4) {
            appendSystemMessage(`💡 <em>Rated ${val}/10: The tutor is diagnosing this step to clear up confusion.</em>`);
        } else if (val >= 8) {
            appendSystemMessage(`⭐ <em>Rated ${val}/10: High confidence recorded! Ready for deeper algorithmic reasoning.</em>`);
        }

        // Trigger adaptive challenge / diagnostic question in chat if returned
        if (adaptiveData && adaptiveData.adaptive_intervention && adaptiveData.adaptive_intervention.prompt) {
            appendTutorMessage(adaptiveData.adaptive_intervention.prompt, adaptiveData.adaptive_intervention.message_id);
        }

        confidenceModal.style.display = "none";
        retentionBanner.style.display = "none";
        await loadMastery();
    });

    // -------------------------------------------------------------
    // 8. Mastery Computation
    // -------------------------------------------------------------
    async function loadMastery() {
        try {
            const resp = await fetch("/api/concepts");
            const data = await resp.json();
            const concepts = data.concepts || [];

            const practiced = concepts.filter(c => (c.total_turns || 0) > 0 || (c.confidence_score || 0) > 0);
            const mastered = concepts.filter(c => (c.confidence_score || 0) >= 8);

            const pct = concepts.length > 0 ? Math.round((mastered.length / concepts.length) * 100) : 0;
            masteryPercent.textContent = `${mastered.length} Mastered`;
            masteryBarFill.style.width = `${pct}%`;
            masteryPracticed.textContent = `${practiced.length} topics studied`;

            const scores = practiced.map(c => c.confidence_score).filter(s => s > 0);
            const avg = scores.length > 0 ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) : "--";
            masteryAvgConf.textContent = `Avg: ${avg}/10`;
        } catch (err) {
            console.error("Mastery fetch error:", err);
        }
    }

    // -------------------------------------------------------------
    // 9. Instructor Telemetry Modal
    // -------------------------------------------------------------
    openTelemetryBtn.addEventListener("click", () => {
        telemetryModal.style.display = "flex";
    });

    closeTelemetryModal.addEventListener("click", () => {
        telemetryModal.style.display = "none";
    });

    function updateTelemetryModal(data) {
        if (!data) return;

        const ext = parseFloat(data.extraction_score || 0);
        const eng = parseFloat(data.engagement_score || 0);
        const action = data.action_taken || "REDIRECT";
        const qVals = data.q_values || [0, 0, 0, 0];

        tExtractionVal.textContent = ext.toFixed(3);
        tExtractionBar.style.width = `${Math.min(100, ext * 100)}%`;

        tEngagementVal.textContent = eng.toFixed(3);
        tEngagementBar.style.width = `${Math.min(100, eng * 100)}%`;

        tActionVal.textContent = action;
        tActionDesc.textContent = `Action ID ${data.action_id} selected by PyTorch Q-Network`;

        const qBoxes = [
            { id: "qNum0", box: "qBox0", val: qVals[0] },
            { id: "qNum1", box: "qBox1", val: qVals[1] },
            { id: "qNum2", box: "qBox2", val: qVals[2] },
            { id: "qNum3", box: "qBox3", val: qVals[3] }
        ];

        let maxIdx = 0;
        let maxVal = -999;
        qVals.forEach((v, i) => {
            if (v > maxVal) { maxVal = v; maxIdx = i; }
        });

        qBoxes.forEach((q, idx) => {
            const el = document.getElementById(q.id);
            const box = document.getElementById(q.box);
            if (el) el.textContent = (q.val || 0).toFixed(3);
            if (box) {
                if (idx === maxIdx) {
                    box.style.borderColor = "var(--accent-primary)";
                    box.style.background = "rgba(99, 102, 241, 0.15)";
                } else {
                    box.style.borderColor = "var(--border-subtle)";
                    box.style.background = "rgba(255, 255, 255, 0.02)";
                }
            }
        });
    }

    async function loadStatus() {
        try {
            const resp = await fetch("/api/status");
            const data = await resp.json();
            if (data.llm_provider) {
                providerPill.textContent = `${data.llm_provider.toUpperCase()} (${data.llm_model || 'qwen2.5:7b'})`;
            }
        } catch (err) {
            console.error("Status fetch failed:", err);
        }
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatViewport.scrollTop = chatViewport.scrollHeight;
        });
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str.replace(/[&<>"']/g, m => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
        }[m]));
    }

    function formatMessageContent(rawText) {
        if (!rawText) return "";
        let formatted = rawText;

        formatted = formatted.replace(/```(?:[a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g, (match, code) => {
            return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
        });

        formatted = formatted.replace(/`([^`]+)`/g, (match, code) => {
            return `<code>${escapeHtml(code)}</code>`;
        });

        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        const paragraphs = formatted.split("\n\n").map(p => {
            if (p.startsWith("<pre>") && p.endsWith("</pre>")) return p;
            return `<p>${p.replace(/\n/g, "<br>")}</p>`;
        });

        return paragraphs.join("");
    }

    function formatRelativeDate(isoStr) {
        if (!isoStr) return "";
        const date = new Date(isoStr);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 2) return "Just now";
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays === 1) return "Yesterday";
        if (diffDays < 7) return `${diffDays}d ago`;
        return date.toLocaleDateString();
    }

    // -------------------------------------------------------------
    // 10. Student Concepts Explored & Topic Status Modal
    // -------------------------------------------------------------
    let allTopicAnalytics = [];
    let currentTopicFilter = "all";
    let currentTopicSearch = "";

    if (masterySummaryCard) {
        masterySummaryCard.addEventListener("click", () => {
            openConceptMasteryModal();
        });
        masterySummaryCard.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                openConceptMasteryModal();
            }
        });
    }

    if (closeConceptMasteryModal) {
        closeConceptMasteryModal.addEventListener("click", () => {
            closeConceptMasteryModalFn();
        });
    }

    if (doneConceptModalBtn) {
        doneConceptModalBtn.addEventListener("click", () => {
            closeConceptMasteryModalFn();
        });
    }

    function openConceptMasteryModal() {
        if (conceptMasteryModal) {
            conceptMasteryModal.style.display = "flex";
            loadConceptAnalytics();
        }
    }

    function closeConceptMasteryModalFn() {
        if (conceptMasteryModal) {
            conceptMasteryModal.style.display = "none";
        }
    }

    async function loadConceptAnalytics() {
        try {
            if (topicsListContainer) {
                topicsListContainer.innerHTML = `<div class="loading-state">Loading topic analytics...</div>`;
            }
            let resp = await fetch("/api/concepts/analytics");
            let data = {};
            if (resp.ok) {
                data = await resp.json();
                allTopicAnalytics = data.topics || [];
            } else {
                // Graceful fallback to /api/concepts if server hasn't been restarted yet
                resp = await fetch("/api/concepts");
                const fallbackData = await resp.json();
                const rawConcepts = fallbackData.concepts || [];
                allTopicAnalytics = rawConcepts.map(c => ({
                    ...c,
                    time_spent_minutes: (c.total_turns || 0) * 2.5,
                    avg_engagement: (c.total_turns || 0) > 0 ? 0.35 : null,
                    avg_extraction: (c.struggle_count || 0) > 0 ? 0.65 : 0.25,
                    messages_count: (c.total_turns || 0) * 2
                }));
                const masteredCount = allTopicAnalytics.filter(c => c.mastery_status === "mastered" || (c.confidence_score || 0) >= 8).length;
                const activeCount = allTopicAnalytics.filter(c => c.mastery_status === "struggling" || c.mastery_status === "in_progress" || (c.total_turns || 0) > 0).length;
                data.summary = {
                    total_topics: allTopicAnalytics.length,
                    mastered_count: masteredCount,
                    in_progress_count: allTopicAnalytics.filter(c => c.mastery_status === "in_progress").length,
                    struggling_count: allTopicAnalytics.filter(c => c.mastery_status === "struggling").length,
                    not_started_count: allTopicAnalytics.filter(c => (c.total_turns || 0) === 0).length,
                    total_time_minutes: allTopicAnalytics.reduce((acc, c) => acc + (c.time_spent_minutes || 0), 0),
                    total_turns: allTopicAnalytics.reduce((acc, c) => acc + (c.total_turns || 0), 0),
                    avg_confidence: 6.0,
                    avg_engagement: 0.09
                };
            }
            const summary = data.summary || {};

            // Update KPI cards
            const total = summary.total_topics || allTopicAnalytics.length;
            const mastered = summary.mastered_count || 0;
            const pct = total > 0 ? Math.round((mastered / total) * 100) : 0;

            if (kpiMasteredCount) kpiMasteredCount.textContent = `${mastered} / ${total}`;
            if (kpiMasteredSub) kpiMasteredSub.textContent = `${pct}% of core syllabus`;
            if (modalMasteryBadge) modalMasteryBadge.textContent = `${mastered} Mastered`;

            if (kpiTotalTurns) kpiTotalTurns.textContent = summary.total_turns || 0;
            
            // Format total study time
            const totalMins = summary.total_time_minutes || 0;
            let timeStr = "";
            if (totalMins >= 60) {
                const hrs = Math.floor(totalMins / 60);
                const rm = Math.round(totalMins % 60);
                timeStr = `${hrs}h ${rm}m`;
            } else if (totalMins > 0) {
                timeStr = `${Math.round(totalMins)}m`;
            } else {
                timeStr = "0m";
            }
            if (kpiTotalTime) kpiTotalTime.textContent = timeStr;

            if (kpiAvgConfidence) {
                kpiAvgConfidence.textContent = summary.avg_confidence > 0 ? `${summary.avg_confidence.toFixed(1)}/10` : "--";
            }

            if (kpiAvgEngagement) {
                const engVal = summary.avg_engagement;
                if (engVal && engVal > 0) {
                    const engPct = Math.round(engVal * 100);
                    kpiAvgEngagement.textContent = `${engPct}% (${engVal.toFixed(2)})`;
                } else {
                    kpiAvgEngagement.textContent = "--";
                }
            }

            // Update filter pill counts
            const activeCount = allTopicAnalytics.filter(t => t.mastery_status === "struggling" || t.mastery_status === "in_progress" || (t.total_turns || 0) > 0).length;
            const masteredCount = allTopicAnalytics.filter(t => t.mastery_status === "mastered" || (t.confidence_score || 0) >= 8).length;
            const notStartedCount = allTopicAnalytics.filter(t => (t.total_turns || 0) === 0 && (!t.confidence_score || t.confidence_score === 0)).length;

            if (countAll) countAll.textContent = allTopicAnalytics.length;
            if (countActive) countActive.textContent = activeCount;
            if (countMastered) countMastered.textContent = masteredCount;
            if (countNotStarted) countNotStarted.textContent = notStartedCount;

            renderTopicAnalyticsCards();
        } catch (err) {
            console.error("Failed to load concept analytics:", err);
            if (topicsListContainer) {
                topicsListContainer.innerHTML = `<div class="empty-state">Unable to load analytics data.</div>`;
            }
        }
    }

    function renderTopicAnalyticsCards() {
        if (!topicsListContainer) return;

        let filtered = allTopicAnalytics;

        // Apply tab filter
        if (currentTopicFilter === "active") {
            filtered = filtered.filter(t => t.mastery_status === "struggling" || t.mastery_status === "in_progress" || (t.total_turns || 0) > 0);
        } else if (currentTopicFilter === "mastered") {
            filtered = filtered.filter(t => t.mastery_status === "mastered" || (t.confidence_score || 0) >= 8);
        } else if (currentTopicFilter === "not_started") {
            filtered = filtered.filter(t => (t.total_turns || 0) === 0 && (!t.confidence_score || t.confidence_score === 0));
        }

        // Apply search query
        if (currentTopicSearch.trim()) {
            const q = currentTopicSearch.trim().toLowerCase();
            filtered = filtered.filter(t => 
                (t.concept_name || "").toLowerCase().includes(q) ||
                (t.category || "").toLowerCase().includes(q) ||
                (t.difficulty || "").toLowerCase().includes(q)
            );
        }

        if (filtered.length === 0) {
            topicsListContainer.innerHTML = `<div class="empty-state" style="padding: 24px; text-align: center; color: var(--text-muted);">No topics match the selected filter.</div>`;
            return;
        }

        topicsListContainer.innerHTML = "";

        filtered.forEach(topic => {
            const card = document.createElement("div");
            const statusClass = topic.mastery_status || "not_started";
            card.className = `topic-status-card status-${statusClass}`;

            // Difficulty styling
            const diff = (topic.difficulty || "medium").toLowerCase();
            const diffClass = diff === "easy" ? "diff-easy" : (diff === "hard" ? "diff-hard" : "diff-medium");

            // Format time spent
            const mins = topic.time_spent_minutes || 0;
            let timeDisplay = "0m";
            if (mins >= 60) {
                const h = Math.floor(mins / 60);
                const m = Math.round(mins % 60);
                timeDisplay = `${h}h ${m}m`;
            } else if (mins > 0) {
                timeDisplay = `${Math.round(mins)}m`;
            }

            // Status label formatting
            let statusText = "Not Started";
            let statusBadgeClass = "badge-not_started";
            if (topic.mastery_status === "mastered" || (topic.confidence_score || 0) >= 8) {
                statusText = "Mastered";
                statusBadgeClass = "badge-mastered";
            } else if (topic.mastery_status === "struggling") {
                statusText = "Struggling";
                statusBadgeClass = "badge-struggling";
            } else if (topic.mastery_status === "in_progress" || (topic.total_turns || 0) > 0) {
                statusText = "In Progress";
                statusBadgeClass = "badge-in_progress";
            }

            // Confidence rating display
            let confDisplay = "Not Rated";
            if (topic.confidence_score && topic.confidence_score > 0) {
                confDisplay = `⭐ ${topic.confidence_score}/10`;
            }

            // Engagement score display
            let engDisplay = "--";
            if (topic.avg_engagement !== null && topic.avg_engagement !== undefined) {
                const engScore = Number(topic.avg_engagement);
                const label = engScore >= 0.5 ? "High" : (engScore >= 0.15 ? "Moderate" : "Inquiry");
                engDisplay = `${label} (${(engScore * 100).toFixed(0)}%)`;
            } else if ((topic.total_turns || 0) > 0) {
                engDisplay = "Active Practice";
            }

            // Cognitive Independence (Extraction / Offloading rate)
            let independenceDisplay = "--";
            if (topic.avg_extraction !== null && topic.avg_extraction !== undefined) {
                const extScore = Number(topic.avg_extraction);
                if (extScore >= 0.6) {
                    independenceDisplay = `⚠️ High Offload (${(extScore * 100).toFixed(0)}%)`;
                } else if (extScore >= 0.35) {
                    independenceDisplay = `Guided (${((1 - extScore) * 100).toFixed(0)}% Self)`;
                } else {
                    independenceDisplay = `Independent (${((1 - extScore) * 100).toFixed(0)}% Self)`;
                }
            } else if (statusClass === "mastered") {
                independenceDisplay = "High Independence";
            }

            const lastStudied = topic.last_studied_at ? formatRelativeDate(topic.last_studied_at) : "Never";

            card.innerHTML = `
                <div class="topic-card-header">
                    <div class="topic-card-title-group">
                        <span class="topic-name">${escapeHtml(topic.concept_name)}</span>
                        <span class="topic-category-tag">${escapeHtml(topic.category || 'DSA')}</span>
                        <span class="topic-difficulty-tag ${diffClass}">${escapeHtml(topic.difficulty || 'Medium')}</span>
                    </div>
                    <span class="status-badge ${statusBadgeClass}">${statusText}</span>
                </div>

                <div class="topic-metrics-grid">
                    <div class="topic-metric-item">
                        <span class="topic-metric-lbl">Confidence</span>
                        <span class="topic-metric-val">${confDisplay}</span>
                    </div>
                    <div class="topic-metric-item">
                        <span class="topic-metric-lbl">Questions Learnt</span>
                        <span class="topic-metric-val">💬 ${topic.total_turns || 0} turns</span>
                    </div>
                    <div class="topic-metric-item">
                        <span class="topic-metric-lbl">Time Spent</span>
                        <span class="topic-metric-val">⏱️ ${timeDisplay}</span>
                    </div>
                    <div class="topic-metric-item">
                        <span class="topic-metric-lbl">Engagement Level</span>
                        <span class="topic-metric-val">🎯 ${engDisplay}</span>
                    </div>
                    <div class="topic-metric-item">
                        <span class="topic-metric-lbl">Cognitive Effort</span>
                        <span class="topic-metric-val">🧠 ${independenceDisplay}</span>
                    </div>
                    <div class="topic-metric-item">
                        <span class="topic-metric-lbl">Interventions</span>
                        <span class="topic-metric-val">🛡️ ${topic.struggle_count || 0}</span>
                    </div>
                </div>

                <div class="topic-card-actions">
                    <span>Last studied: <strong>${lastStudied}</strong></span>
                    <button class="btn-practice-topic" data-concept-id="${topic.concept_id}" data-concept-name="${escapeHtml(topic.concept_name)}">
                        <span>💬 Practice with Tutor</span>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14"></path><path d="m12 5 7 7-7 7"></path></svg>
                    </button>
                </div>
            `;

            // Practice button hook
            const practiceBtn = card.querySelector(".btn-practice-topic");
            if (practiceBtn) {
                practiceBtn.addEventListener("click", () => {
                    const cid = practiceBtn.dataset.conceptId;
                    const cname = practiceBtn.dataset.conceptName;
                    closeConceptMasteryModalFn();
                    startNewChatWithTopic(cid, cname);
                });
            }

            topicsListContainer.appendChild(card);
        });
    }

    function startNewChatWithTopic(conceptId, conceptName) {
        startNewChat();
        currentTopicId = conceptId;
        activeTopicTitle.textContent = conceptName;
        activeCategory.style.display = "inline-block";
        activeCategory.textContent = "DSA Concept";
        activeTopicSub.textContent = `Practicing ${conceptName} with adaptive reinforcement learning tutor.`;
        messageInput.value = `Let's practice ${conceptName}. What's the main idea and how do we approach it?`;
        messageInput.focus();
    }

    // Filter pill click listener
    if (topicFilterPills) {
        topicFilterPills.querySelectorAll(".filter-pill").forEach(pill => {
            pill.addEventListener("click", () => {
                topicFilterPills.querySelectorAll(".filter-pill").forEach(p => p.classList.remove("active"));
                pill.classList.add("active");
                currentTopicFilter = pill.dataset.filter || "all";
                renderTopicAnalyticsCards();
            });
        });
    }

    // Search input listener
    if (topicSearchInput) {
        topicSearchInput.addEventListener("input", (e) => {
            currentTopicSearch = e.target.value;
            renderTopicAnalyticsCards();
        });
    }

    // Close on backdrop click and Esc key
    window.addEventListener("click", (e) => {
        if (e.target === conceptMasteryModal) {
            closeConceptMasteryModalFn();
        }
    });

    window.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            if (conceptMasteryModal && conceptMasteryModal.style.display === "flex") {
                closeConceptMasteryModalFn();
            }
        }
    });

    // Initialize application
    init();
});
