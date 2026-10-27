/**
 * Adaptive DSA Tutor — Client Application Logic
 * Communicates with Flask backend and updates ML Diagnostics in real-time.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatContainer = document.getElementById("chatContainer");
    const messageInput = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");
    const resetBtn = document.getElementById("resetBtn");
    const policyToggle = document.getElementById("policyToggle");
    const modeLabel = document.getElementById("modeLabel");
    const problemSelect = document.getElementById("problemSelect");
    const toggleProblemBtn = document.getElementById("toggleProblemBtn");
    const probBody = document.getElementById("probBody");

    // Problem Meta Elements
    const probTitle = document.getElementById("probTitle");
    const probDifficulty = document.getElementById("probDifficulty");
    const probCategory = document.getElementById("probCategory");
    const probDesc = document.getElementById("probDesc");
    const probExample = document.getElementById("probExample");

    // Diagnostics HUD Elements
    const sessionCounter = document.getElementById("sessionCounter");
    const actionDisplay = document.getElementById("actionDisplay");
    const actionIcon = document.getElementById("actionIcon");
    const actionName = document.getElementById("actionName");
    const actionDesc = document.getElementById("actionDesc");
    const actionCard = document.querySelector(".action-card");
    const actionMethod = document.getElementById("actionMethod");

    const extractionVal = document.getElementById("extractionVal");
    const extractionBar = document.getElementById("extractionBar");
    const engagementVal = document.getElementById("engagementVal");
    const engagementBar = document.getElementById("engagementBar");
    const struggleStatus = document.getElementById("struggleStatus");
    const struggleDots = document.querySelectorAll(".s-dot");

    // Action Descriptions Dictionary
    const ACTION_INFO = {
        "REDIRECT": {
            icon: "🛑",
            title: "REDIRECT",
            theme: "theme-REDIRECT",
            desc: "Active cognitive offloading detected. Refused direct code dump; redirected student to step 1."
        },
        "SCAFFOLD": {
            icon: "🪜",
            title: "SCAFFOLD",
            theme: "theme-SCAFFOLD",
            desc: "Student is experiencing cognitive overload or confusion. Deconstructing problem into smaller sub-steps."
        },
        "SOCRATIC": {
            icon: "💡",
            title: "SOCRATIC",
            theme: "theme-SOCRATIC",
            desc: "Student is reasoning productively. Guiding discovery via targeted probing questions without giving answers."
        },
        "DELAY": {
            icon: "⏳",
            title: "DELAY",
            theme: "theme-DELAY",
            desc: "Student fatigue / burnout detected. Prescribed a 10-minute strategic pause while preserving problem state."
        },
        "BASELINE": {
            icon: "🤖",
            title: "BASELINE (GENERIC)",
            theme: "",
            desc: "Standard unconstrained LLM response (non-adaptive baseline comparison)."
        }
    };

    let problemsData = [];
    let isSending = false;

    // -------------------------------------------------------------
    // 1. Initialize Problem Bank
    // -------------------------------------------------------------
    async function loadProblems() {
        try {
            const resp = await fetch("/api/problems");
            const data = await resp.json();
            problemsData = data.problems || [];

            problemSelect.innerHTML = "";
            problemsData.forEach((prob, idx) => {
                const opt = document.createElement("option");
                opt.value = prob.id;
                opt.textContent = `${prob.difficulty}: ${prob.title}`;
                if (idx === 0) opt.selected = true;
                problemSelect.appendChild(opt);
            });

            if (problemsData.length > 0) {
                renderProblem(problemsData[0]);
            }
        } catch (err) {
            console.error("Failed to load problems:", err);
        }
    }

    function renderProblem(prob) {
        probTitle.textContent = prob.title;
        probDifficulty.textContent = prob.difficulty;
        probCategory.textContent = prob.category;
        probDesc.textContent = prob.description;
        probExample.textContent = prob.example;

        // Difficulty badge color
        if (prob.difficulty === "Easy") {
            probDifficulty.style.color = "var(--accent-emerald)";
            probDifficulty.style.borderColor = "rgba(16, 185, 129, 0.4)";
        } else if (prob.difficulty === "Medium") {
            probDifficulty.style.color = "var(--accent-amber)";
            probDifficulty.style.borderColor = "rgba(245, 158, 11, 0.4)";
        } else {
            probDifficulty.style.color = "var(--accent-rose)";
            probDifficulty.style.borderColor = "rgba(244, 63, 94, 0.4)";
        }
    }

    problemSelect.addEventListener("change", (e) => {
        const found = problemsData.find(p => p.id === e.target.value);
        if (found) {
            renderProblem(found);
            // Announce in chat
            appendSystemMessage(`📌 Switched problem to: <strong>${found.title}</strong> (${found.category})`);
        }
    });

    // Toggle problem drawer
    toggleProblemBtn.addEventListener("click", () => {
        if (probBody.style.display === "none") {
            probBody.style.display = "flex";
            toggleProblemBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>`;
        } else {
            probBody.style.display = "none";
            toggleProblemBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="6 15 12 9 18 15"></polyline>
                </svg>`;
        }
    });

    // -------------------------------------------------------------
    // 2. Mode Toggle (Policy vs Baseline)
    // -------------------------------------------------------------
    policyToggle.addEventListener("change", (e) => {
        const isPolicy = e.target.checked;
        modeLabel.textContent = isPolicy ? "Mode: Adaptive RL" : "Mode: Baseline (Generic)";
        actionMethod.textContent = isPolicy ? "RL Policy Active" : "Baseline Mode";
        appendSystemMessage(isPolicy ? 
            "⚡ <strong>Adaptive RL Policy Mode Activated.</strong> The system will monitor extraction and intervene with Q-learning actions." : 
            "🤖 <strong>Baseline Mode Activated.</strong> The tutor will answer like a standard LLM without anti-offloading interventions."
        );
    });

    // -------------------------------------------------------------
    // 3. Quick Test Chips
    // -------------------------------------------------------------
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const promptText = chip.getAttribute("data-prompt");
            if (promptText && !isSending) {
                messageInput.value = promptText;
                sendMessage();
            }
        });
    });

    // -------------------------------------------------------------
    // 4. Send Message & Receive Inference
    // -------------------------------------------------------------
    async function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || isSending) return;

        isSending = true;
        sendBtn.disabled = true;
        messageInput.value = "";
        messageInput.style.height = "auto";

        // Append Student Message
        appendStudentMessage(text);

        // Show Typing Indicator
        const typingId = appendTypingIndicator();

        try {
            const usePolicy = policyToggle.checked;
            const resp = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text, use_policy: usePolicy })
            });

            const data = await resp.json();
            removeTypingIndicator(typingId);

            if (data.error) {
                appendSystemMessage(`⚠️ Error: ${data.error}`);
            } else {
                appendTutorMessage(data.response, data.action_taken, data.is_fallback);
                updateDiagnostics(data);
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

    // Auto-grow textarea
    messageInput.addEventListener("input", function() {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
    });

    // -------------------------------------------------------------
    // 5. Message Rendering & Helpers
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
        chatContainer.appendChild(row);
        scrollToBottom();
    }

    function appendTutorMessage(text, actionName, isFallback) {
        const row = document.createElement("div");
        row.className = "message-row tutor-row";

        const badgeClass = `badge-${actionName}`;
        const fallbackNote = isFallback ? '<span title="Template fallback used"> (Fallback)</span>' : '';

        row.innerHTML = `
            <div class="avatar tutor-avatar">🎓</div>
            <div class="message-bubble tutor-bubble">
                <div class="tutor-bubble-header">
                    <span class="bubble-action-badge ${badgeClass}">${actionName}</span>
                    <span class="bubble-meta">Adaptive Tutor${fallbackNote}</span>
                </div>
                <div class="message-text">${formatMessageContent(text)}</div>
            </div>
        `;
        chatContainer.appendChild(row);
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
        chatContainer.appendChild(row);
        scrollToBottom();
    }

    function appendTypingIndicator() {
        const id = "typing-" + Date.now();
        const row = document.createElement("div");
        row.id = id;
        row.className = "message-row tutor-row";
        row.innerHTML = `
            <div class="avatar tutor-avatar">🎓</div>
            <div class="message-bubble tutor-bubble">
                <div class="typing-skeleton">
                    <div class="dot-pulse"></div>
                    <div class="dot-pulse"></div>
                    <div class="dot-pulse"></div>
                </div>
            </div>
        `;
        chatContainer.appendChild(row);
        scrollToBottom();
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    function formatMessageContent(raw) {
        if (!raw) return "";

        // Check for code blocks ```lang ... ```
        const codeBlockRegex = /```([a-zA-Z]*)\n([\s\S]*?)```/g;
        let formatted = raw.replace(codeBlockRegex, (match, lang, code) => {
            return `<pre><code class="language-${lang}">${escapeHtml(code.trim())}</code></pre>`;
        });

        // Inline code `...`
        formatted = formatted.replace(/`([^`]+)`/g, (match, code) => {
            return `<code>${escapeHtml(code)}</code>`;
        });

        // Split paragraphs
        const paragraphs = formatted.split(/\n\s*\n/);
        return paragraphs.map(p => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
    }

    // -------------------------------------------------------------
    // 6. Update Real-Time ML Diagnostics HUD
    // -------------------------------------------------------------
    function updateDiagnostics(data) {
        const action = data.action_taken || "REDIRECT";
        const info = ACTION_INFO[action] || ACTION_INFO["REDIRECT"];

        // Update Action Card
        actionIcon.textContent = info.icon;
        actionName.textContent = info.title;
        actionDesc.textContent = info.desc;

        // Apply Card Theme
        actionCard.className = "hud-card action-card " + info.theme;

        // Update Session & Turn Info
        const turnNum = data.turn_number || 1;
        const struggle = data.struggle_turns || 0;
        sessionCounter.textContent = `Turn ${turnNum} • Struggle: ${struggle}`;

        // Update Extraction Gauge
        const extrProb = Math.min(Math.max(data.extraction_score || 0, 0), 1);
        const extrPct = (extrProb * 100).toFixed(1);
        extractionVal.textContent = `${extrPct}%`;
        extractionBar.style.width = `${extrPct}%`;

        // Update Engagement Gauge
        const engScore = Math.min(Math.max(data.engagement_score || 0, 0), 1);
        const engPct = (engScore * 100).toFixed(1);
        engagementVal.textContent = `${engPct}%`;
        engagementBar.style.width = `${engPct}%`;

        // Update Struggle Dots
        struggleDots.forEach((dot, idx) => {
            if (idx < struggle) {
                dot.className = "s-dot active";
            } else {
                dot.className = "s-dot";
            }
        });

        if (struggle === 0) {
            struggleStatus.textContent = "Refreshed";
            struggleStatus.style.color = "var(--accent-emerald)";
        } else if (struggle < 3) {
            struggleStatus.textContent = "Cognitive Load Normal";
            struggleStatus.style.color = "var(--accent-amber)";
        } else {
            struggleStatus.textContent = "Fatigue / Pause Advised";
            struggleStatus.style.color = "var(--action-delay)";
        }

        // Update Q-Values Bar Graph
        const qVals = data.q_values || [0, 0, 0, 0];
        const maxQ = Math.max(...qVals);
        const minQ = Math.min(...qVals);
        const range = (maxQ - minQ) || 1.0;

        qVals.forEach((q, idx) => {
            const valEl = document.getElementById(`qVal-${idx}`);
            const fillEl = document.getElementById(`qFill-${idx}`);
            const groupEl = document.getElementById(`qGroup-${idx}`);

            if (valEl) valEl.textContent = q >= 0 ? `+${q.toFixed(4)}` : q.toFixed(4);

            // Normalized visual percentage for bar width (min 15% for visibility)
            let widthPct = 15;
            if (maxQ > minQ) {
                widthPct = 15 + ((q - minQ) / range) * 80;
            } else if (maxQ > 0) {
                widthPct = 60;
            }
            if (fillEl) fillEl.style.width = `${widthPct}%`;

            // Highlight winning action
            if (groupEl) {
                if (idx === data.action_id && data.action_taken !== "BASELINE") {
                    groupEl.classList.add("winning-action");
                } else {
                    groupEl.classList.remove("winning-action");
                }
            }
        });
    }

    // -------------------------------------------------------------
    // 7. Reset Conversation
    // -------------------------------------------------------------
    resetBtn.addEventListener("click", async () => {
        try {
            await fetch("/api/reset", { method: "POST" });
            chatContainer.innerHTML = "";

            // Re-render welcome card
            const welcomeCard = document.createElement("div");
            welcomeCard.className = "welcome-card";
            welcomeCard.innerHTML = `
                <div class="welcome-icon">🎓</div>
                <h3>Session Reset & Refreshed</h3>
                <p>Cognitive struggle counters and conversation context have been reset to Turn 0.</p>
                <div class="feature-pills">
                    <span class="pill pill-redirect">🛑 REDIRECT (Anti-Shortcut)</span>
                    <span class="pill pill-scaffold">🪜 SCAFFOLD (Deconstruct)</span>
                    <span class="pill pill-socratic">💡 SOCRATIC (Inquire)</span>
                    <span class="pill pill-delay">⏳ DELAY (Anti-Burnout)</span>
                </div>
            `;
            chatContainer.appendChild(welcomeCard);

            // Reset HUD
            sessionCounter.textContent = "Turn 0 • Struggle: 0";
            extractionVal.textContent = "0.0%";
            extractionBar.style.width = "0%";
            engagementVal.textContent = "0.0%";
            engagementBar.style.width = "0%";
            struggleDots.forEach(d => d.className = "s-dot");
            struggleStatus.textContent = "Refreshed";
            struggleStatus.style.color = "var(--accent-emerald)";

            actionCard.className = "hud-card action-card";
            actionIcon.textContent = "⚡";
            actionName.textContent = "AWAITING INPUT";
            actionDesc.textContent = "Send a student prompt to observe RL intervention selection.";

            [0, 1, 2, 3].forEach(idx => {
                const valEl = document.getElementById(`qVal-${idx}`);
                const fillEl = document.getElementById(`qFill-${idx}`);
                const groupEl = document.getElementById(`qGroup-${idx}`);
                if (valEl) valEl.textContent = "0.0000";
                if (fillEl) fillEl.style.width = "0%";
                if (groupEl) groupEl.classList.remove("winning-action");
            });

            appendSystemMessage("🔄 <strong>Session successfully reset.</strong> Ready for fresh student interaction.");
        } catch (err) {
            console.error("Reset failed:", err);
        }
    });

    // Kick off initialization
    loadProblems();
});
