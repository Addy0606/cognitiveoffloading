"""
FastAPI Application for Adaptive DSA Tutor
Serves the web platform and streams real-time RL telemetry logs to the terminal.

Run with:
    python fastapi_app.py
OR
    uvicorn fastapi_app:app --reload --port 5000
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.adaptive_tutor import AdaptiveTutor
from src.memory_store import MemoryStore
from src.adaptive_questions import get_adaptive_intervention
from src.utils import setup_logging

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger("dsa_tutor_fastapi")

app = FastAPI(
    title="DSA AI Mentor API",
    description="Adaptive DSA Learning Platform with RL Anti-Offloading Policy",
    version="2.0.0"
)

# Mount static folder
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")

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
    directly to the terminal as requests arrive.
    """
    sep = "=" * 70
    q_formatted = " | ".join([
        f"RED: {q_vals[0]:.3f}",
        f"SCAF: {q_vals[1]:.3f}",
        f"SOC: {q_vals[2]:.3f}",
        f"DEL: {q_vals[3]:.3f}"
    ]) if len(q_vals) == 4 else str(q_vals)

    print("\n" + sep)
    print(f" [FASTAPI TELEMETRY] Session: {conv_id} | Topic: {topic_name or 'General DSA'}")
    print(sep)
    print(f"  [STUDENT MSG]      \"{student_msg[:80]}{'...' if len(student_msg) > 80 else ''}\"")
    print(f"  [EXTRACTION PROB]  {extraction_score:.3f} {'(OFFLOADING DETECTED - CODE WITHHELD)' if extraction_score > 0.5 else '(GENUINE REASONING EFFORT)'}")
    print(f"  [ENGAGEMENT SCORE] {engagement_score:.3f}")
    if action_id == -2:
        print(f"  [RL ACTION CHOSEN] {action_taken} (Social/Conversational Turn — RL Intervention Bypassed)")
    else:
        print(f"  [RL ACTION CHOSEN] {action_taken} (Action ID: {action_id})")
    print(f"  [TUTOR RESPONSE]   \"{tutor_reply[:100].replace(chr(10), ' ')}{'...' if len(tutor_reply) > 100 else ''}\"")
    print(sep + "\n")


# -------------------------------------------------------------
# Request Models
# -------------------------------------------------------------
class CreateConvRequest(BaseModel):
    topic_id: Optional[str] = None
    topic_name: Optional[str] = None
    title: Optional[str] = None

class RenameConvRequest(BaseModel):
    title: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    topic_id: Optional[str] = None
    use_policy: bool = True
    active_dwell_seconds: Optional[float] = 0.0

class FeedbackRequest(BaseModel):
    message_id: Optional[int] = None
    rating: Optional[int] = None
    label: Optional[str] = None
    concept_id: Optional[str] = None
    conversation_id: Optional[str] = None

class HeartbeatRequest(BaseModel):
    conversation_id: Optional[str] = None
    topic_id: Optional[str] = None
    active_seconds: float = 15.0

class ResolveCheckinRequest(BaseModel):
    concept_id: str
    confidence_score: int = 7
    notes: Optional[str] = ""


# -------------------------------------------------------------
# Web Page Route
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = ROOT_DIR / "templates" / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


# -------------------------------------------------------------
# Conversations Endpoints
# -------------------------------------------------------------
@app.get("/api/conversations")
async def get_conversations():
    conversations = memory.get_conversations()
    return {"conversations": conversations}

@app.post("/api/conversations")
async def create_conversation(req: CreateConvRequest):
    topic_name = req.topic_name
    if req.topic_id and not topic_name:
        concept = memory.get_concept(req.topic_id)
        if concept:
            topic_name = concept["concept_name"]

    conv_id = memory.create_conversation(topic_id=req.topic_id, topic_name=topic_name, title=req.title)
    return {
        "status": "created",
        "conversation_id": conv_id,
        "title": req.title or (f"{topic_name}" if topic_name else "New Chat"),
        "topic_id": req.topic_id,
        "topic_name": topic_name
    }

@app.get("/api/conversations/{conv_id}")
async def get_conversation_details(conv_id: str):
    conv = memory.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = memory.get_messages(conv_id)
    topic = memory.get_concept(conv["topic_id"]) if conv.get("topic_id") else None

    return {
        "conversation": conv,
        "topic": topic,
        "messages": messages
    }

@app.put("/api/conversations/{conv_id}/title")
@app.patch("/api/conversations/{conv_id}/title")
async def rename_conversation(conv_id: str, req: RenameConvRequest):
    new_title = req.title.strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")

    memory.rename_conversation(conv_id, new_title)
    return {
        "status": "renamed",
        "conversation_id": conv_id,
        "title": new_title
    }

@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    memory.delete_conversation(conv_id)
    return {"status": "deleted", "conversation_id": conv_id}


# -------------------------------------------------------------
# Retention & Checkin Endpoints
# -------------------------------------------------------------
@app.get("/api/checkin")
async def get_retention_checkin(exclude_topic_id: Optional[str] = None):
    checkin = memory.get_retention_checkin(exclude_concept_id=exclude_topic_id)
    return {"checkin": checkin}

@app.post("/api/checkin/resolve")
async def resolve_checkin(req: ResolveCheckinRequest):
    memory.update_concept_confidence(req.concept_id, req.confidence_score, req.notes or "")
    return {"status": "resolved", "concept_id": req.concept_id, "confidence": req.confidence_score}


# -------------------------------------------------------------
# Student Feedback Endpoint
# -------------------------------------------------------------
@app.post("/api/feedback")
async def save_feedback(req: FeedbackRequest):
    res = memory.save_feedback(
        message_id=req.message_id,
        rating=req.rating,
        label=req.label,
        concept_id=req.concept_id
    )

    # Actively adjust tutor state and policy based on student feedback
    if (req.rating is not None and req.rating <= 4) or req.label == "confused":
        tutor.struggle_turns += 2
        logger.info(f"Student gave low confidence feedback ({req.rating or req.label}) -> struggle_turns escalated to {tutor.struggle_turns}")
    elif (req.rating is not None and req.rating >= 7) or req.label == "clear":
        tutor.struggle_turns = max(0, tutor.struggle_turns - 1)
        logger.info(f"Student gave high confidence feedback ({req.rating or req.label}) -> struggle_turns decayed to {tutor.struggle_turns}")

    # Generate adaptive intervention question based on feedback rating
    adaptive_intervention = None
    if req.concept_id:
        concept = memory.get_concept(req.concept_id)
        c_name = concept["concept_name"] if concept else None
        adaptive_intervention = get_adaptive_intervention(
            concept_id=req.concept_id,
            rating=req.rating,
            label=req.label,
            concept_name=c_name
        )

        # If conversation_id is provided, also insert this adaptive check as a tutor message in the conversation!
        if req.conversation_id:
            adaptive_msg = adaptive_intervention["prompt"]
            adaptive_msg_id = memory.add_message(
                conv_id=req.conversation_id,
                role="tutor",
                content=adaptive_msg,
                action_taken="ADAPTIVE_CHECK"
            )
            adaptive_intervention["message_id"] = adaptive_msg_id

    res["adaptive_intervention"] = adaptive_intervention
    return res


# -------------------------------------------------------------
# Active Focus Heartbeat Endpoint (ACET)
# -------------------------------------------------------------
@app.post("/api/heartbeat")
async def record_heartbeat(req: HeartbeatRequest):
    memory.record_active_time(
        concept_id=req.topic_id,
        conv_id=req.conversation_id,
        active_seconds=req.active_seconds
    )
    return {"status": "ok", "active_seconds": req.active_seconds}


# -------------------------------------------------------------
# Chat Endpoint
# -------------------------------------------------------------
@app.post("/api/chat")
async def chat(req: ChatRequest):
    message = req.message.strip()
    conv_id = req.conversation_id
    topic_id = req.topic_id
    use_policy = req.use_policy

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Ensure conversation exists
    if not conv_id:
        conv_id = memory.create_conversation(topic_id=topic_id)
        conv = memory.get_conversation(conv_id)
    else:
        conv = memory.get_conversation(conv_id)
        if not conv:
            conv_id = memory.create_conversation(topic_id=topic_id)
            conv = memory.get_conversation(conv_id)

    # Auto concept detection & mid-chat topic transition
    detected_concept = memory.detect_concept(message)
    topic_switched = False
    old_topic_name = conv.get("topic_name") or ""
    old_topic_id = conv.get("topic_id")

    # If the conversation did not have a topic_id explicitly set, infer from the title or prior messages
    if not old_topic_id and conv and conv.get("title"):
        inferred = memory.detect_concept(conv.get("title", ""))
        if inferred:
            old_topic_id = inferred["concept_id"]
            old_topic_name = inferred["concept_name"]

    if detected_concept:
        if old_topic_id and detected_concept["concept_id"] != old_topic_id:
            # Topic switch detected mid-conversation!
            topic_switched = True
            topic_id = detected_concept["concept_id"]
            with memory._get_connection() as conn:
                conn.execute(
                    "UPDATE conversations SET topic_id = ?, topic_name = ? WHERE id = ?",
                    (topic_id, detected_concept["concept_name"], conv_id)
                )
                conn.commit()
            # Reset tutor struggle counter so previous topic struggle doesn't bleed into new topic
            tutor.struggle_turns = 0
            tutor.is_post_break = True
            logger.info(f"Topic switched mid-chat from '{old_topic_name}' to '{detected_concept['concept_name']}'")
        elif not topic_id:
            topic_id = detected_concept["concept_id"]
            with memory._get_connection() as conn:
                conn.execute(
                    "UPDATE conversations SET topic_id = ?, topic_name = ? WHERE id = ?",
                    (topic_id, detected_concept["concept_name"], conv_id)
                )
                conn.commit()

    # Smart auto-title
    updated_title = None
    if conv and conv.get("title") in ("New Chat", "New DSA Session", None):
        auto_title = memory.auto_title(message, detected_concept)
        memory.rename_conversation(conv_id, auto_title)
        updated_title = auto_title

    # Retrieve history
    history_messages = memory.get_messages(conv_id)
    formatted_history = [
        {"role": "user" if m["role"] == "student" else "assistant", "message": m["content"]}
        for m in history_messages[-8:]
    ]

    # Topic context
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

    # Student learning context & feedback adaptation
    student_context = ""
    if topic_switched and old_topic_name:
        student_context += (
            f"🚨 CRITICAL TOPIC TRANSITION DIRECTIVE:\n"
            f"The student was previously working on '{old_topic_name}', and has now switched to '{topic_name}'.\n"
            f"In your VERY FIRST SENTENCE, you MUST acknowledge the transition and ask for their feedback on '{old_topic_name}':\n"
            f"e.g. 'Before we shift to {topic_name}, how comfortable do you feel setting {old_topic_name} aside? Did everything click for you?'\n"
            f"Then, proceed to address their new question about '{topic_name}'.\n\n"
        )
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

        # Print rich telemetry directly to the terminal!
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

        # If topic transition occurred, persist the interactive transition card in conversation history
        transition_msg_id = None
        if topic_switched and old_topic_id and old_topic_name:
            transition_msg_id = memory.add_topic_transition(
                conv_id=conv_id,
                old_topic_id=old_topic_id,
                old_topic_name=old_topic_name,
                new_topic_name=topic_name
            )

        # Determine if inline comprehension feedback should be shown on this message turn:
        # Show ONLY on:
        # 1. Student expresses resolution ("thanks", "got it", "i get it", "understood", "makes sense", "solved")
        # 2. Recovery turn after low confidence / confusion (struggle_turns > 0)
        # 3. Topic practice milestone (every 4 student turns)
        msg_lower = message.lower()
        is_resolution = any(w in msg_lower for w in ["thank", "got it", "i get it", "makes sense", "understood", "i understand", "now i get", "clear now", "solved", "working now"])
        is_post_struggle = (tutor.struggle_turns > 0 or getattr(tutor, "is_post_break", False))
        topic_turn_count = len([m for m in history_messages if m.get("role") == "student"])
        is_milestone = (topic_turn_count > 0 and topic_turn_count % 4 == 0)
        should_show_feedback = bool(is_resolution or is_post_struggle or is_milestone)

        # Bounded cognitive dwell time (5s - 180s per turn)
        dwell_secs = max(5.0, min(float(req.active_dwell_seconds or 45.0), 180.0))

        # Save to SQLite
        memory.add_message(
            conv_id=conv_id,
            role="student",
            content=message,
            dwell_time_seconds=dwell_secs
        )
        tutor_msg_id = memory.add_message(
            conv_id=conv_id,
            role="tutor",
            content=tutor_response,
            action_taken=action_taken,
            extraction_score=extraction_score,
            engagement_score=engagement_score,
            q_values=q_vals,
            meta_json=json.dumps({"show_feedback": should_show_feedback})
        )

        # Update concept activity, authentic active time & struggle
        effective_concept_id = topic_id or (detected_concept["concept_id"] if detected_concept else None)
        if effective_concept_id:
            memory.touch_concept(effective_concept_id, active_seconds=dwell_secs)
            if extraction_score > 0.6 or action_taken in ("REDIRECT", "DELAY"):
                memory.record_concept_struggle(effective_concept_id, 1)

        return {
            "response": tutor_response,
            "message_id": tutor_msg_id,
            "conversation_id": conv_id,
            "title": updated_title,
            "topic_id": effective_concept_id,
            "topic_name": topic_name,
            "topic_switched": topic_switched,
            "old_topic_id": old_topic_id if topic_switched else None,
            "old_topic_name": old_topic_name if topic_switched else None,
            "transition_msg_id": transition_msg_id,
            "show_feedback": should_show_feedback,
            "action_taken": action_taken,
            "action_id": action_id,
            "extraction_score": extraction_score,
            "engagement_score": engagement_score,
            "q_values": q_vals,
            "is_fallback": bool(result.get("is_fallback", False)),
            "llm_provider": str(result.get("llm_provider", "ollama"))
        }

    except Exception as e:
        logger.exception("Error during tutor inference")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/concepts")
async def get_concepts():
    concepts = memory.get_all_concepts()
    return {"concepts": concepts}

@app.get("/api/concepts/analytics")
async def get_concepts_analytics():
    return memory.get_concept_analytics()

@app.get("/api/status")
async def get_status():
    return {
        "server": "FastAPI (Uvicorn)",
        "llm_provider": tutor.llm_provider,
        "llm_model": tutor.llm_model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        "classifier_loaded": classifier_path.exists(),
        "policy_loaded": policy_path.exists(),
        "database": str(memory.db_path)
    }




if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 68)
    print("  [*] ADAPTIVE DSA TUTOR — FASTAPI SERVER")
    print(f"  [*] URL: http://127.0.0.1:{port}")
    print("  [*] Swagger Docs: http://127.0.0.1:5000/docs")
    print("  [*] Telemetry: Real-time RL inference logs streamed below")
    print("=" * 68 + "\n")
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=port, reload=True)
