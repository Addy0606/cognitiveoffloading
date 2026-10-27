"""
Flask Web Application for Adaptive DSA Tutor
Provides a modern visual interface for interacting with the RL-driven tutoring system.
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.adaptive_tutor import AdaptiveTutor
from src.utils import setup_logging, set_seed

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

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

# Sample DSA Problem Bank for Demo
DSA_PROBLEMS = [
    {
        "id": "reverse_linked_list",
        "title": "Reverse a Linked List",
        "difficulty": "Easy",
        "category": "Linked Lists",
        "description": "Given the head of a singly linked list, reverse the list, and return the reversed list.",
        "example": "Input: head = [1,2,3,4,5]  ->  Output: [5,4,3,2,1]",
        "constraints": "Number of nodes is [0, 5000]. -5000 <= Node.val <= 5000"
    },
    {
        "id": "binary_search",
        "title": "Binary Search",
        "difficulty": "Easy",
        "category": "Searching",
        "description": "Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, return its index. Otherwise, return -1.",
        "example": "Input: nums = [-1,0,3,5,9,12], target = 9  ->  Output: 4",
        "constraints": "1 <= nums.length <= 10^4. All integers are unique. nums is sorted."
    },
    {
        "id": "knapsack_dp",
        "title": "0/1 Knapsack Problem",
        "difficulty": "Medium",
        "category": "Dynamic Programming",
        "description": "Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value in the knapsack. You cannot break an item (either take complete item or don't take it).",
        "example": "Input: W = 50, values = [60, 100, 120], weights = [10, 20, 30]  ->  Output: 220",
        "constraints": "1 <= n <= 1000, 1 <= W <= 1000"
    },
    {
        "id": "cycle_detection",
        "title": "Linked List Cycle Detection",
        "difficulty": "Medium",
        "category": "Two Pointers",
        "description": "Given head, the head of a linked list, determine if the linked list has a cycle in it using Floyd's Cycle-Finding Algorithm (Tortoise and Hare).",
        "example": "Input: head = [3,2,0,-4], pos = 1  ->  Output: true",
        "constraints": "Number of nodes is in [0, 10^4]. Space complexity must be O(1)."
    },
    {
        "id": "graph_bfs",
        "title": "Shortest Path in Unweighted Graph",
        "difficulty": "Medium",
        "category": "Graphs / BFS",
        "description": "Given an unweighted graph, a source vertex s, and destination vertex d, find the shortest path from s to d using Breadth-First Search.",
        "example": "Input: V = 5, E = [[0,1],[0,2],[1,3],[2,3],[3,4]], s = 0, d = 4  ->  Output: Shortest distance = 3",
        "constraints": "Graph may contain cycles. Must handle disconnected components."
    }
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/problems", methods=["GET"])
def get_problems():
    return jsonify({"problems": DSA_PROBLEMS})

@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "llm_provider": tutor.llm_provider,
        "llm_model": tutor.llm_model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        "classifier_loaded": classifier_path.exists(),
        "policy_loaded": policy_path.exists(),
        "turn_number": tutor.turn_number,
        "struggle_turns": tutor.struggle_turns,
        "history_length": len(tutor.conversation_history) // 2
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    use_policy = data.get("use_policy", True)

    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        result = tutor.infer(message, use_policy=use_policy)
        
        # Clean up numpy / tensor types for clean JSON response
        q_vals = [float(q) for q in result.get("q_values", [0.0, 0.0, 0.0, 0.0])]
        
        return jsonify({
            "response": result.get("tutor_response", ""),
            "action_taken": result.get("action_taken", "REDIRECT"),
            "action_id": int(result.get("action_id", 0)),
            "extraction_score": float(result.get("extraction_score", 0.0)),
            "engagement_score": float(result.get("engagement_score", 0.0)),
            "engagement_breakdown": result.get("engagement_breakdown", {}),
            "q_values": q_vals,
            "turn_number": int(tutor.turn_number),
            "struggle_turns": int(tutor.struggle_turns),
            "is_fallback": bool(result.get("is_fallback", False)),
            "llm_provider": str(result.get("llm_provider", "ollama"))
        })
    except Exception as e:
        logger.exception("Error during tutor inference")
        return jsonify({"error": str(e)}), 500

@app.route("/api/reset", methods=["POST"])
def reset():
    tutor.reset_conversation()
    logger.info("Conversation reset via API")
    return jsonify({
        "status": "reset",
        "turn_number": tutor.turn_number,
        "struggle_turns": tutor.struggle_turns
    })

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 65)
    print("  [*] ADAPTIVE DSA TUTOR WEB UI")
    print(f"  [*] URL: http://127.0.0.1:{port}")
    print("  [*] Press Ctrl+C to stop the server")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
