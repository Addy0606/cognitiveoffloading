"""
Flask Web Application for Adaptive DSA Tutor
Platform for DSA learning with:
- SQLite persistence across sessions, days, and reboots
- Freeform chats with smart auto-naming and manual renaming
- Automatic concept detection & spaced retention check-ins
- Pedagogical RL policy & anti-cognitive offloading detection
- Student confidence feedback (1-10)
- Detailed server terminal telemetry logs
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.adaptive_tutor import AdaptiveTutor
from src.memory_store import MemoryStore
from src.utils import setup_logging, set_seed

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger("dsa_tutor")

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize SQLite Memory Store
memory = MemoryStore()

# Initialize Tutor instance
logger.info("Initializing Adaptive Tutor...")
tutor = AdaptiveTutor()

# Load models if present
classifier_path = ROOT_DIR / "models" / "classifier_v1"
policy_path = ROOT_DIR / "models" / "policy_network_v1.pth"

if classifier_path.exists():
    tutor.classifier.load(str(classifier_path))
    logger.info(f"Loaded classifier from {classifier_path}")
else:
    logger.warning("Classifier not found. Initializing untrained backbone.")
    tutor.classifier.initialize()

if policy_path.exists():
    tutor.rl_trainer.load(str(policy_path))
    logger.info(f"Loaded policy from {policy_path}")
else:
    logger.warning("Policy checkpoint not found.")


def log_telemetry_to_terminal(
    student_msg: str,
    extraction_score: float,
    engagement_score: float,
    action_taken: str,
    action_id: int,
    q_vals: list,
    tutor_reply: str,
    conv_id: str,
    topic_name: str = ""
):
    """
    Print clear, structured server-side logs for instructors/evaluators
    without cluttering the student's learning interface.
    """
    sep = "=" * 70
    q_formatted = " | ".join([
        f"RED: {q_vals[0]:.3f}",
        f"SCAF: {q_vals[1]:.3f}",
        f"SOC: {q_vals[2]:.3f}",
        f"DEL: {q_vals[3]:.3f}"
    ]) if len(q_vals) == 4 else str(q_vals)

    print("\n" + sep)
    print(f" [DSA TUTOR TELEMETRY] Session: {conv_id} | Topic: {topic_name or 'General DSA'}")
    print(sep)
    print(f"  [STUDENT MSG]      \"{student_msg[:80]}{'...' if len(student_msg) > 80 else ''}\"")
    print(f"  [EXTRACTION PROB]  {extraction_score:.3f} {'(OFFLOADING / DIRECT CODE DEMAND)' if extraction_score > 0.5 else '(GENUINE REASONING EFFORT)'}")
    print(f"  [ENGAGEMENT SCORE] {engagement_score:.3f}")
    print(f"  [Q-VALUES]         [{q_formatted}]")
    print(f"  [RL ACTION CHOSEN] {action_taken} (Action ID: {action_id})")
    print(f"  [TUTOR RESPONSE]   \"{tutor_reply[:100].replace(chr(10), ' ')}{'...' if len(tutor_reply) > 100 else ''}\"")
    print(sep + "\n")


@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------------------------------------
# Conversations API (Persistence, Renaming, Deletion)
# -------------------------------------------------------------
@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    """List all stored conversations with metadata."""
    conversations = memory.get_conversations()
    return jsonify({"conversations": conversations})


@app.route("/api/conversations", methods=["POST"])
def create_conversation():
    """Create a new chat conversation session."""
    data = request.get_json() or {}
    topic_id = data.get("topic_id")
    topic_name = data.get("topic_name")
    title = data.get("title")

    if topic_id and not topic_name:
        concept = memory.get_concept(topic_id)
        if concept:
            topic_name = concept["concept_name"]

    conv_id = memory.create_conversation(topic_id=topic_id, topic_name=topic_name, title=title)
    return jsonify({
        "status": "created",
        "conversation_id": conv_id,
        "title": title or (f"{topic_name}" if topic_name else "New Chat"),
        "topic_id": topic_id,
        "topic_name": topic_name
    })


@app.route("/api/conversations/<conv_id>", methods=["GET"])
def get_conversation_details(conv_id):
    """Retrieve full conversation history."""
    conv = memory.get_conversation(conv_id)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404

    messages = memory.get_messages(conv_id)
    topic = memory.get_concept(conv["topic_id"]) if conv.get("topic_id") else None

    return jsonify({
        "conversation": conv,
        "topic": topic,
        "messages": messages
    })


@app.route("/api/conversations/<conv_id>/title", methods=["PUT", "PATCH"])
def rename_conversation(conv_id):
    """Manually rename a conversation."""
    data = request.get_json() or {}
    new_title = data.get("title", "").strip()
    if not new_title:
        return jsonify({"error": "Title cannot be empty"}), 400

    memory.rename_conversation(conv_id, new_title)
    return jsonify({
        "status": "renamed",
        "conversation_id": conv_id,
        "title": new_title
    })


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    """Delete a conversation."""
    memory.delete_conversation(conv_id)
    return jsonify({"status": "deleted", "conversation_id": conv_id})


# -------------------------------------------------------------
# Retention & Spaced Review API (Teacher Requirement 3)
# -------------------------------------------------------------
@app.route("/api/checkin", methods=["GET"])
def get_retention_checkin():
    """
    Returns spaced retention checkin candidate.
    Only returns a concept if the student actually practiced it in a previous session!
    """
    exclude_id = request.args.get("exclude_topic_id")
    checkin = memory.get_retention_checkin(exclude_concept_id=exclude_id)
    return jsonify({"checkin": checkin})


@app.route("/api/checkin/resolve", methods=["POST"])
def resolve_checkin():
    """Update student confidence after a check-in."""
    data = request.get_json() or {}
    concept_id = data.get("concept_id")
    confidence = data.get("confidence_score", 7)
    notes = data.get("notes", "")

    if not concept_id:
        return jsonify({"error": "concept_id is required"}), 400

    memory.update_concept_confidence(concept_id, int(confidence), notes)
    return jsonify({"status": "resolved", "concept_id": concept_id, "confidence": confidence})


# -------------------------------------------------------------
# Student Feedback API (Teacher Requirement 3)
# -------------------------------------------------------------
@app.route("/api/feedback", methods=["POST"])
def save_feedback():
    """Save student confidence rating (1-10) or comprehension label on tutor message."""
    data = request.get_json() or {}
    message_id = data.get("message_id")
    rating = data.get("rating")
    label = data.get("label")
    concept_id = data.get("concept_id")

    if not message_id:
        return jsonify({"error": "message_id is required"}), 400

    res = memory.save_feedback(
        message_id=int(message_id),
        rating=int(rating) if rating is not None else None,
        label=label,
        concept_id=concept_id
    )

    # Actively adjust tutor state and policy based on student feedback
    if (rating is not None and rating <= 4) or label == "confused":
        tutor.struggle_turns += 2
        logger.info(f"Student gave low confidence rating ({rating or label}) -> struggle_turns escalated to {tutor.struggle_turns}")
    elif (rating is not None and rating >= 7) or label == "clear":
        tutor.struggle_turns = max(0, tutor.struggle_turns - 1)
        logger.info(f"Student gave high confidence rating ({rating or label}) -> struggle_turns decayed to {tutor.struggle_turns}")

    return jsonify(res)


# -------------------------------------------------------------
# Chat Endpoint (Auto Concept Detection + Auto Titling)
# -------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    """Process student message through RL policy and save to memory store."""
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    conv_id = data.get("conversation_id")
    topic_id = data.get("topic_id")
    use_policy = data.get("use_policy", True)

    if not message:
        return jsonify({"error": "Message is required"}), 400

    # Ensure a conversation exists
    if not conv_id:
        conv_id = memory.create_conversation(topic_id=topic_id)
        conv = memory.get_conversation(conv_id)
    else:
        conv = memory.get_conversation(conv_id)
        if not conv:
            conv_id = memory.create_conversation(topic_id=topic_id)
            conv = memory.get_conversation(conv_id)

    # Automatic concept detection from student query
    detected_concept = memory.detect_concept(message)
    if detected_concept and not topic_id:
        topic_id = detected_concept["concept_id"]
        # Update conversation topic
        with memory._get_connection() as conn:
            conn.execute(
                "UPDATE conversations SET topic_id = ?, topic_name = ? WHERE id = ?",
                (topic_id, detected_concept["concept_name"], conv_id)
            )
            conn.commit()

    # Smart auto-title if conversation has placeholder title "New Chat"
    updated_title = None
    if conv and conv.get("title") in ("New Chat", "New DSA Session", None):
        auto_title = memory.auto_title(message, detected_concept)
        memory.rename_conversation(conv_id, auto_title)
        updated_title = auto_title

    # Retrieve prior conversation turns from SQLite for multi-turn context
    history_messages = memory.get_messages(conv_id)
    formatted_history = [
        {"role": "user" if m["role"] == "student" else "assistant", "message": m["content"]}
        for m in history_messages[-8:]
    ]

    # Build topic context
    topic_context = ""
    topic_name = ""
    active_concept = memory.get_concept(topic_id) if topic_id else detected_concept
    if active_concept:
        topic_name = active_concept["concept_name"]
        topic_context = (
            f"Concept / Problem: {active_concept['concept_name']} ({active_concept.get('category', 'DSA')})\n"
            f"Description: {active_concept.get('description', '')}\n"
            f"Example: {active_concept.get('example', '')}"
        )

    # Build spaced retention student context & active feedback adaptation
    student_context = ""
    if history_messages:
        last_tutor_msg = next((m for m in reversed(history_messages) if m["role"] == "tutor"), None)
        if last_tutor_msg:
            r = last_tutor_msg.get("feedback_rating")
            lbl = last_tutor_msg.get("feedback_label")
            if (r is not None and r <= 4) or lbl == "confused":
                student_context += (
                    f"STUDENT COMPREHENSION ALERT: The student marked their understanding as {r or 'Low'}/10 (Confused). "
                    "The previous explanation was not clear enough for them. "
                    "You MUST adjust your approach: break down the idea into simpler sub-steps, avoid technical jargon, "
                    "use intuitive visual analogies, and guide them with gentle scaffolding.\n\n"
                )
            elif (r is not None and r >= 8) or lbl == "clear":
                student_context += (
                    f"The student understood the previous step well (Rating: {r or 'Clear'}/10). "
                    "You can move forward to the next logical concept or ask a deeper reasoning question.\n\n"
                )

    checkin = memory.get_retention_checkin(exclude_concept_id=topic_id)
    if checkin:
        student_context += (
            f"The student previously studied '{checkin['concept_name']}' with confidence "
            f"{checkin['confidence_score']}/10. If the student asks about past concepts or returns, "
            f"you may reference this background smoothly."
        )

    try:
        # Run inference
        result = tutor.infer(
            student_message=message,
            conversation_history=formatted_history,
            use_policy=use_policy,
            topic_context=topic_context,
            student_context=student_context,
        )

        q_vals = [float(q) for q in result.get("q_values", [0.0, 0.0, 0.0, 0.0])]
        action_taken = result.get("action_taken", "REDIRECT")
        action_id = int(result.get("action_id", 0))
        extraction_score = float(result.get("extraction_score", 0.0))
        engagement_score = float(result.get("engagement_score", 0.0))
        tutor_response = result.get("tutor_response", "")

        # Log server telemetry to terminal
        log_telemetry_to_terminal(
            student_msg=message,
            extraction_score=extraction_score,
            engagement_score=engagement_score,
            action_taken=action_taken,
            action_id=action_id,
            q_vals=q_vals,
            tutor_reply=tutor_response,
            conv_id=conv_id,
            topic_name=topic_name
        )

        # 1. Save student message to SQLite
        student_msg_id = memory.add_message(
            conv_id=conv_id,
            role="student",
            content=message
        )

        # 2. Save tutor response to SQLite
        tutor_msg_id = memory.add_message(
            conv_id=conv_id,
            role="tutor",
            content=tutor_response,
            action_taken=action_taken,
            extraction_score=extraction_score,
            engagement_score=engagement_score,
            q_values=q_vals
        )

        # 3. Update concept activity & struggle state
        effective_concept_id = topic_id or (detected_concept["concept_id"] if detected_concept else None)
        if effective_concept_id:
            memory.touch_concept(effective_concept_id)
            if extraction_score > 0.6 or action_taken in ("REDIRECT", "DELAY"):
                memory.record_concept_struggle(effective_concept_id, 1)

        return jsonify({
            "response": tutor_response,
            "message_id": tutor_msg_id,
            "conversation_id": conv_id,
            "title": updated_title,
            "topic_id": effective_concept_id,
            "topic_name": topic_name,
            "action_taken": action_taken,
            "action_id": action_id,
            "extraction_score": extraction_score,
            "engagement_score": engagement_score,
            "q_values": q_vals,
            "is_fallback": bool(result.get("is_fallback", False)),
            "llm_provider": str(result.get("llm_provider", "ollama"))
        })

    except Exception as e:
        logger.exception("Error during tutor inference")
        return jsonify({"error": str(e)}), 500


@app.route("/api/concepts", methods=["GET"])
def get_concepts():
    """Return all concepts with student mastery status."""
    concepts = memory.get_all_concepts()
    return jsonify({"concepts": concepts})


@app.route("/api/concepts/analytics", methods=["GET"])
def get_concepts_analytics():
    """Return all concepts with aggregated student analytics."""
    return jsonify(memory.get_concept_analytics())


@app.route("/api/status", methods=["GET"])
def get_status():
    """System health check and loaded models status."""
    return jsonify({
        "llm_provider": tutor.llm_provider,
        "llm_model": tutor.llm_model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        "classifier_loaded": classifier_path.exists(),
        "policy_loaded": policy_path.exists(),
        "database": str(memory.db_path)
    })


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 68)
    print("  [*] ADAPTIVE DSA TUTOR — STUDENT LEARNING PLATFORM")
    print(f"  [*] URL: http://127.0.0.1:{port}")
    print("  [*] Memory Store: data/tutor_memory.db (SQLite Persistent)")
    print("  [*] Press Ctrl+C to stop the server")
    print("=" * 68 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
