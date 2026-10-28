"""
Memory and Conversation Persistence Store for Adaptive DSA Tutor.
Provides SQLite-based long-term persistence across sessions, days, and reboots:
- Conversation and message history with auto-naming & manual renaming
- Automatic concept detection from student queries
- Student knowledge state & confidence ratings (1-10)
- Spaced retrieval / retention check-ins (e.g. following up on AVL Trees taught yesterday)
"""

import sqlite3
import uuid
import json
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "tutor_memory.db"

# Core curated DSA concepts for mastery tracking
DEFAULT_CONCEPTS = [
    {
        "id": "avl_trees",
        "name": "AVL Trees & Self-Balancing",
        "category": "Trees",
        "difficulty": "Hard",
        "description": "Self-balancing binary search trees using balance factor calculation and Single/Double rotations (LL, RR, LR, RL).",
        "example": "Inserting 10, 20, 30 triggers an RR rotation to keep tree height O(log n)."
    },
    {
        "id": "binary_search_tree",
        "name": "Binary Search Trees (BST)",
        "category": "Trees",
        "difficulty": "Medium",
        "description": "Hierarchical data structure with the property that left children are smaller and right children are larger.",
        "example": "Inorder traversal of a BST yields elements in sorted ascending order."
    },
    {
        "id": "reverse_linked_list",
        "name": "Reverse a Linked List",
        "category": "Linked Lists",
        "difficulty": "Easy",
        "description": "Reverse node pointer directions in a singly linked list iteratively or recursively in O(n) time and O(1) space.",
        "example": "Input: 1 -> 2 -> 3 -> None  =>  Output: 3 -> 2 -> 1 -> None"
    },
    {
        "id": "cycle_detection",
        "name": "Linked List Cycle Detection",
        "category": "Linked Lists",
        "difficulty": "Medium",
        "description": "Detect whether a linked list contains a cycle using Floyd's Tortoise and Hare algorithm.",
        "example": "Slow pointer moves 1 step, fast pointer moves 2 steps; intersection proves a cycle exists."
    },
    {
        "id": "binary_search",
        "name": "Binary Search",
        "category": "Searching",
        "difficulty": "Easy",
        "description": "Search target element in a sorted collection by repeatedly halving the search space in O(log n).",
        "example": "nums = [-1, 0, 3, 5, 9, 12], target = 9 => index 4"
    },
    {
        "id": "two_pointers",
        "name": "Two Pointers Technique",
        "category": "Searching",
        "difficulty": "Easy",
        "description": "Use two index pointers moving toward each other or at varying speeds to search or filter linear data.",
        "example": "Two Sum II in sorted array: left starts at 0, right starts at n-1."
    },
    {
        "id": "knapsack_dp",
        "name": "0/1 Knapsack Problem",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": "Maximize total value without exceeding capacity W where each item can only be taken once.",
        "example": "Weights = [10, 20, 30], Values = [60, 100, 120], W = 50 => Max Value = 220"
    },
    {
        "id": "coin_change",
        "name": "Coin Change (Min Coins)",
        "category": "Dynamic Programming",
        "difficulty": "Medium",
        "description": "Find minimum number of coins needed to make amount using available denominations (unbounded DP).",
        "example": "coins = [1, 2, 5], amount = 11 => 3 coins (5 + 5 + 1)"
    },
    {
        "id": "graph_bfs",
        "name": "Shortest Path (Graph BFS)",
        "category": "Graphs",
        "difficulty": "Medium",
        "description": "Find shortest path between nodes in an unweighted graph using queue-based level order traversal.",
        "example": "Nodes visited level-by-level guarantee shortest path distance."
    },
    {
        "id": "dijkstra",
        "name": "Dijkstra's Shortest Path",
        "category": "Graphs",
        "difficulty": "Medium",
        "description": "Single-source shortest path algorithm for non-negative weighted graphs using min-heaps in O((V + E) log V).",
        "example": "Greedily relax nearest unvisited vertex until all vertices are settled."
    },
    {
        "id": "valid_parentheses",
        "name": "Valid Parentheses (Stack)",
        "category": "Stacks & Queues",
        "difficulty": "Easy",
        "description": "Determine if brackets '()', '{}', '[]' in a string are closed in correct nested order using LIFO stack.",
        "example": "Input: '()[]{}' => true; Input: '(]' => false"
    }
]

# Keywords mapping for automatic concept detection from freeform queries
CONCEPT_KEYWORDS = {
    "avl_trees": ["avl", "self-balancing", "balance factor", "rr rotation", "ll rotation", "lr rotation", "rl rotation", "tree rotation", "balanced bst"],
    "binary_search_tree": [
        "binary tree", "binary trees", "bst", "binary search tree", "binary search trees",
        "inorder traversal", "in-order", "preorder traversal", "pre-order",
        "postorder traversal", "post-order", "tree traversal", "mirror image of binary tree",
        "mirror tree", "invert binary tree", "tree insert", "tree delete", "tree height", "root node"
    ],
    "reverse_linked_list": [
        "reverse linked list", "reverse a linked list", "reverse list", "reverse node",
        "reverse pointers", "reversing a linked list", "reverse singly linked list",
        "linked list", "linked lists", "singly linked list"
    ],
    "cycle_detection": ["cycle detection", "floyd", "tortoise and hare", "detect cycle", "linked list cycle", "has cycle", "cycle in linked list"],
    "binary_search": ["binary search", "search sorted", "bisect", "log n search", "rotated array", "binary search algorithm"],
    "two_pointers": ["two pointers", "two sum", "pair sum", "opposite ends", "sliding window"],
    "knapsack_dp": ["knapsack", "0/1 knapsack", "knapsack problem", "weight and value", "dynamic programming knapsack"],
    "coin_change": ["coin change", "fewest coins", "min coins", "minimum coins", "coin denominations"],
    "graph_bfs": ["bfs", "breadth first", "shortest path unweighted", "level order"],
    "dijkstra": ["dijkstra", "shortest path weighted", "min-heap shortest path", "relax edge"],
    "valid_parentheses": ["valid parentheses", "matching brackets", "parentheses stack", "balanced brackets", "stack brackets"]
}


class MemoryStore:
    """Manages SQLite storage for chats, messages, and student mastery."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables if they don't exist and pre-populate default concepts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Conversations table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                topic_id TEXT,
                topic_name TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)

            # 2. Messages table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                action_taken TEXT,
                extraction_score REAL,
                engagement_score REAL,
                q_values TEXT,
                feedback_rating INTEGER,
                feedback_label TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
            """)

            # 3. Student Knowledge / Concept Mastery table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_knowledge (
                concept_id TEXT PRIMARY KEY,
                concept_name TEXT NOT NULL,
                category TEXT NOT NULL,
                difficulty TEXT,
                description TEXT,
                example TEXT,
                last_studied_at TEXT,
                total_turns INTEGER DEFAULT 0,
                struggle_count INTEGER DEFAULT 0,
                confidence_score INTEGER DEFAULT 0,
                mastery_status TEXT DEFAULT 'not_started',
                notes TEXT
            )
            """)

            # Ensure all default concepts exist in student_knowledge
            for c in DEFAULT_CONCEPTS:
                cursor.execute("""
                INSERT OR IGNORE INTO student_knowledge 
                (concept_id, concept_name, category, difficulty, description, example, mastery_status, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, 'not_started', 0)
                """, (c["id"], c["name"], c["category"], c["difficulty"], c["description"], c["example"]))

            # Migrations for Active Cognitive Engagement Time (ACET) & persistent topic transitions
            for migration in [
                "ALTER TABLE student_knowledge ADD COLUMN total_time_seconds REAL DEFAULT 0.0",
                "ALTER TABLE conversations ADD COLUMN active_time_seconds REAL DEFAULT 0.0",
                "ALTER TABLE messages ADD COLUMN dwell_time_seconds REAL DEFAULT 0.0",
                "ALTER TABLE messages ADD COLUMN message_type TEXT DEFAULT 'chat'",
                "ALTER TABLE messages ADD COLUMN meta_json TEXT DEFAULT ''"
            ]:
                try:
                    cursor.execute(migration)
                except sqlite3.OperationalError:
                    pass

            conn.commit()

    def reset_all_data(self) -> None:
        """Clear all conversation history and reset knowledge practice states to fresh slate."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM conversations")
            conn.execute("""
            UPDATE student_knowledge 
            SET last_studied_at = NULL,
                total_turns = 0,
                struggle_count = 0,
                confidence_score = 0,
                mastery_status = 'not_started',
                notes = NULL
            """)
            conn.commit()
            logger.info("All conversations and practice history reset to clean slate.")

    # ------------------------------------------------------------------
    # Concept Detection & Auto-Naming API
    # ------------------------------------------------------------------
    def detect_concept(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Analyze student query and automatically match with known DSA concept.
        Returns matching concept dictionary or None.
        """
        if not text:
            return None
        text_lower = text.lower()

        # Check multi-word keywords first, then single-word
        for concept_id, keywords in CONCEPT_KEYWORDS.items():
            for kw in sorted(keywords, key=len, reverse=True):
                # Word boundary check
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, text_lower):
                    return self.get_concept(concept_id)

        return None

    def auto_title(self, student_msg: str, detected_concept: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a concise, human-readable title from student's first question.
        e.g. 'How do RR rotations work in AVL tree' -> 'AVL Trees: RR Rotations'
        """
        clean = student_msg.strip().replace("\n", " ")
        # Remove common filler phrases
        clean = re.sub(r'^(can you|please|just|how to|how do|explain|tell me|give me|what is|what are|help me with)\s+', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'^(the|a|an)\s+', '', clean, flags=re.IGNORECASE)

        if detected_concept:
            concept_name = detected_concept["concept_name"]
            # If student asked something specific within the concept
            words = clean.split()[:4]
            snippet = " ".join(words).title()
            if snippet.lower() not in concept_name.lower():
                return f"{concept_name}: {snippet}"[:40]
            return concept_name

        words = clean.split()[:5]
        title = " ".join(words).strip()
        if not title:
            return "DSA Question"
        return (title[0].upper() + title[1:])[:35]

    # ------------------------------------------------------------------
    # Conversations API
    # ------------------------------------------------------------------
    def create_conversation(
        self,
        topic_id: Optional[str] = None,
        topic_name: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """Create a new conversation session and return its ID."""
        conv_id = f"conv_{uuid.uuid4().hex[:10]}"
        now = datetime.now().isoformat()
        final_title = title or (f"{topic_name}" if topic_name else "New Chat")

        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO conversations (id, title, topic_id, topic_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (conv_id, final_title, topic_id, topic_name, now, now))
            conn.commit()

        if topic_id:
            self.touch_concept(topic_id)

        return conv_id

    def get_conversations(self) -> List[Dict[str, Any]]:
        """Return all conversations ordered by latest activity."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT c.*, 
                   COUNT(m.id) as message_count,
                   (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1) as last_message
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def rename_conversation(self, conv_id: str, new_title: str) -> bool:
        """Manually rename a conversation."""
        now = datetime.now().isoformat()
        clean_title = new_title.strip() or "DSA Chat"
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?
            """, (clean_title, now, conv_id))
            conn.commit()
            return True

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and its messages."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            conn.commit()
            return True

    # ------------------------------------------------------------------
    # Messages API
    # ------------------------------------------------------------------
    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        action_taken: str = "",
        extraction_score: float = 0.0,
        engagement_score: float = 0.0,
        q_values: Optional[List[float]] = None,
        dwell_time_seconds: float = 0.0,
        message_type: str = "chat",
        meta_json: str = ""
    ) -> int:
        """Append message to conversation and update updated_at."""
        now = datetime.now().isoformat()
        qv_json = json.dumps(q_values) if q_values else ""

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO messages (
                conversation_id, role, content, action_taken, extraction_score,
                engagement_score, q_values, dwell_time_seconds, message_type, meta_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (conv_id, role, content, action_taken, extraction_score, engagement_score, qv_json, dwell_time_seconds, message_type, meta_json, now))
            msg_id = cursor.lastrowid

            cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
            conn.commit()
            return msg_id

    def add_topic_transition(
        self,
        conv_id: str,
        old_topic_id: str,
        old_topic_name: str,
        new_topic_name: str
    ) -> int:
        """Persist a topic transition interactive card event in conversation history."""
        now = datetime.now().isoformat()
        meta = {
            "old_topic_id": old_topic_id,
            "old_topic_name": old_topic_name,
            "new_topic_name": new_topic_name,
            "resolved": False,
            "rating": None,
            "label": None
        }
        content = f"Topic Transition: Leaving {old_topic_name} → Exploring {new_topic_name}"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO messages (conversation_id, role, content, message_type, meta_json, created_at)
            VALUES (?, 'topic_transition', ?, 'topic_transition', ?, ?)
            """, (conv_id, content, json.dumps(meta), now))
            msg_id = cursor.lastrowid
            conn.commit()
            return msg_id

    def update_transition_feedback(
        self,
        concept_id: str,
        rating: Optional[int] = None,
        label: Optional[str] = None
    ) -> None:
        """Update any pending topic_transition message cards for this concept to mark them resolved."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, meta_json FROM messages WHERE role = 'topic_transition'")
            rows = cursor.fetchall()
            for r in rows:
                if r["meta_json"]:
                    try:
                        meta = json.loads(r["meta_json"])
                        if meta.get("old_topic_id") == concept_id:
                            meta["resolved"] = True
                            if rating is not None:
                                meta["rating"] = rating
                            if label is not None:
                                meta["label"] = label
                            cursor.execute("""
                            UPDATE messages 
                            SET meta_json = ?, feedback_rating = COALESCE(?, feedback_rating), feedback_label = COALESCE(?, feedback_label)
                            WHERE id = ?
                            """, (json.dumps(meta), rating, label, r["id"]))
                    except Exception:
                        pass
            conn.commit()

    def get_messages(self, conv_id: str) -> List[Dict[str, Any]]:
        """Retrieve full message history for a conversation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC
            """, (conv_id,))
            messages = []
            for row in cursor.fetchall():
                d = dict(row)
                if d.get("q_values"):
                    try:
                        d["q_values"] = json.loads(d["q_values"])
                    except Exception:
                        d["q_values"] = []
                if d.get("meta_json"):
                    try:
                        d["meta"] = json.loads(d["meta_json"])
                    except Exception:
                        d["meta"] = {}
                else:
                    d["meta"] = {}
                messages.append(d)
            return messages

    # ------------------------------------------------------------------
    # Feedback & Knowledge State API
    # ------------------------------------------------------------------
    def save_feedback(
        self,
        message_id: Optional[int] = None,
        rating: Optional[int] = None,
        label: Optional[str] = None,
        concept_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record student confidence rating (1-10) or qualitative comprehension tag."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if message_id:
                cursor.execute("""
                UPDATE messages 
                SET feedback_rating = COALESCE(?, feedback_rating),
                    feedback_label = COALESCE(?, feedback_label)
                WHERE id = ?
                """, (rating, label, message_id))

            # If associated with a concept, update student mastery and resolve transition cards
            if concept_id:
                status = "mastered" if (rating and rating >= 8) else ("in_progress" if (rating and rating >= 5) else "struggling")
                cursor.execute("""
                UPDATE student_knowledge
                SET confidence_score = COALESCE(?, confidence_score),
                    mastery_status = ?,
                    last_studied_at = ?
                WHERE concept_id = ?
                """, (rating, status, datetime.now().isoformat(), concept_id))

            conn.commit()

        if concept_id:
            self.update_transition_feedback(concept_id, rating, label)

        return {"status": "success", "message_id": message_id, "rating": rating, "label": label, "concept_id": concept_id}

    def touch_concept(self, concept_id: str, active_seconds: float = 0.0) -> None:
        """
        Mark a concept as active/studied with turn increments and authentic active dwell time.
        Bounded cognitive dwell time: between 5s and 180s per turn to eliminate tab-out inflation.
        """
        now = datetime.now().isoformat()
        capped_seconds = max(5.0, min(float(active_seconds or 45.0), 180.0))
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE student_knowledge 
            SET last_studied_at = ?,
                total_turns = total_turns + 1,
                total_time_seconds = COALESCE(total_time_seconds, 0.0) + ?,
                mastery_status = CASE WHEN mastery_status = 'not_started' THEN 'in_progress' ELSE mastery_status END
            WHERE concept_id = ?
            """, (now, capped_seconds, concept_id))
            conn.commit()

    def record_active_time(
        self,
        concept_id: Optional[str] = None,
        conv_id: Optional[str] = None,
        active_seconds: float = 15.0
    ) -> None:
        """
        Record verified active heartbeat focus time for a concept and conversation.
        Only called when window is visible and active (keystrokes / mouse events).
        Heartbeat increments are bounded to 60s max per ping.
        """
        if not active_seconds or active_seconds <= 0:
            return
        inc = min(float(active_seconds), 60.0)
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            if concept_id:
                conn.execute("""
                UPDATE student_knowledge
                SET total_time_seconds = COALESCE(total_time_seconds, 0.0) + ?,
                    last_studied_at = ?
                WHERE concept_id = ?
                """, (inc, now, concept_id))
            if conv_id:
                conn.execute("""
                UPDATE conversations
                SET active_time_seconds = COALESCE(active_time_seconds, 0.0) + ?,
                    updated_at = ?
                WHERE id = ?
                """, (inc, now, conv_id))
            conn.commit()

    def record_concept_struggle(self, concept_id: str, struggle_inc: int = 1) -> None:
        """Increment struggle counter for a concept."""
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE student_knowledge
            SET struggle_count = struggle_count + ?,
                mastery_status = CASE WHEN confidence_score < 6 THEN 'struggling' ELSE mastery_status END
            WHERE concept_id = ?
            """, (struggle_inc, concept_id))
            conn.commit()

    def update_concept_confidence(self, concept_id: str, confidence: int, notes: str = "") -> None:
        """Directly update a student's confidence rating (1-10) for a concept."""
        status = "mastered" if confidence >= 8 else ("in_progress" if confidence >= 5 else "struggling")
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE student_knowledge
            SET confidence_score = ?,
                mastery_status = ?,
                notes = COALESCE(NULLIF(?, ''), notes),
                last_studied_at = ?
            WHERE concept_id = ?
            """, (confidence, status, notes, now, concept_id))
            conn.commit()

    def get_all_concepts(self) -> List[Dict[str, Any]]:
        """Return all concepts with student mastery status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM student_knowledge ORDER BY category ASC, concept_name ASC")
            return [dict(row) for row in cursor.fetchall()]

    def get_concept_analytics(self) -> Dict[str, Any]:
        """
        Aggregate comprehensive student learning telemetry per topic:
        - confidence score
        - questions / turns learnt
        - struggle interventions count
        - engagement score
        - cognitive offloading (extraction) rate
        - estimated time spent (minutes)
        - last studied at
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM student_knowledge ORDER BY category ASC, concept_name ASC")
            concepts = [dict(row) for row in cursor.fetchall()]

            # Fetch session and message telemetry per topic
            cursor.execute("""
            SELECT 
                c.topic_id,
                c.id as conversation_id,
                MIN(m.created_at) as session_start,
                MAX(m.created_at) as session_end,
                COUNT(m.id) as message_count,
                AVG(CASE WHEN m.engagement_score > 0 THEN m.engagement_score ELSE NULL END) as avg_engagement,
                AVG(CASE WHEN m.extraction_score > 0 THEN m.extraction_score ELSE NULL END) as avg_extraction
            FROM conversations c
            JOIN messages m ON c.id = m.conversation_id
            WHERE c.topic_id IS NOT NULL
            GROUP BY c.topic_id, c.id
            """)
            session_rows = cursor.fetchall()

            # Aggregate per topic_id
            topic_stats = {}
            for r in session_rows:
                tid = r["topic_id"]
                if tid not in topic_stats:
                    topic_stats[tid] = {
                        "total_time_seconds": 0.0,
                        "message_count": 0,
                        "engagement_sum": 0.0,
                        "engagement_count": 0,
                        "extraction_sum": 0.0,
                        "extraction_count": 0,
                    }

                topic_stats[tid]["message_count"] += r["message_count"]

                # Calculate duration in seconds
                if r["session_start"] and r["session_end"]:
                    try:
                        t0 = datetime.fromisoformat(r["session_start"])
                        t1 = datetime.fromisoformat(r["session_end"])
                        duration = (t1 - t0).total_seconds()
                        if duration <= 10 and r["message_count"] > 1:
                            duration = r["message_count"] * 45
                        topic_stats[tid]["total_time_seconds"] += max(duration, 30.0)
                    except Exception:
                        topic_stats[tid]["total_time_seconds"] += r["message_count"] * 45

                if r["avg_engagement"] is not None:
                    topic_stats[tid]["engagement_sum"] += r["avg_engagement"]
                    topic_stats[tid]["engagement_count"] += 1

                if r["avg_extraction"] is not None:
                    topic_stats[tid]["extraction_sum"] += r["avg_extraction"]
                    topic_stats[tid]["extraction_count"] += 1

            enriched = []
            total_time_overall = 0.0
            total_turns_overall = 0
            conf_list = []
            eng_list = []

            for c in concepts:
                cid = c["concept_id"]
                st = topic_stats.get(cid, {})

                # Primary: Authentic recorded active cognitive engagement time (ACET)
                time_secs = float(c.get("total_time_seconds") or 0.0)
                if time_secs == 0:
                    time_secs = float(st.get("total_time_seconds") or 0.0)
                if time_secs == 0 and (c.get("total_turns") or 0) > 0:
                    time_secs = float(c["total_turns"]) * 45.0

                time_minutes = round(time_secs / 60.0, 1)
                total_time_overall += time_secs
                total_turns_overall += (c.get("total_turns") or 0)

                if c.get("confidence_score") and c["confidence_score"] > 0:
                    conf_list.append(c["confidence_score"])

                avg_eng = None
                if st.get("engagement_count", 0) > 0:
                    avg_eng = round(st["engagement_sum"] / st["engagement_count"], 3)
                    eng_list.append(avg_eng)

                avg_ext = None
                if st.get("extraction_count", 0) > 0:
                    avg_ext = round(st["extraction_sum"] / st["extraction_count"], 3)

                c_enriched = {
                    **c,
                    "time_spent_minutes": time_minutes,
                    "avg_engagement": avg_eng,
                    "avg_extraction": avg_ext,
                    "messages_count": st.get("message_count", 0)
                }
                enriched.append(c_enriched)

            mastered_count = sum(1 for c in enriched if c.get("mastery_status") == "mastered" or (c.get("confidence_score") or 0) >= 8)
            struggling_count = sum(1 for c in enriched if c.get("mastery_status") == "struggling")
            in_progress_count = sum(1 for c in enriched if c.get("mastery_status") == "in_progress")
            not_started_count = sum(1 for c in enriched if c.get("mastery_status") in ("not_started", None) and (c.get("total_turns") or 0) == 0)

            overall_summary = {
                "total_topics": len(enriched),
                "mastered_count": mastered_count,
                "in_progress_count": in_progress_count,
                "struggling_count": struggling_count,
                "not_started_count": not_started_count,
                "total_time_minutes": round(total_time_overall / 60.0, 1),
                "total_turns": total_turns_overall,
                "avg_confidence": round(sum(conf_list) / len(conf_list), 1) if conf_list else 0.0,
                "avg_engagement": round(sum(eng_list) / len(eng_list), 3) if eng_list else 0.0,
            }

            return {
                "summary": overall_summary,
                "topics": enriched
            }

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single concept details."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM student_knowledge WHERE concept_id = ?", (concept_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Spaced Retention / Recall Check-In Engine (Teacher Requirement 3)
    # ------------------------------------------------------------------
    def get_retention_checkin(self, exclude_concept_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Check if there is a concept the student ACTUALLY studied in a previous session
        where they struggled or had low confidence (< 7), to ask about it on a new chat.
        Requires total_turns > 0 (prevents synthetic false banners on new installs).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
            SELECT * FROM student_knowledge
            WHERE last_studied_at IS NOT NULL 
              AND total_turns > 0
            """
            params = []
            if exclude_concept_id:
                query += " AND concept_id != ?"
                params.append(exclude_concept_id)

            query += """
            ORDER BY 
                CASE 
                    WHEN mastery_status = 'struggling' THEN 1
                    WHEN confidence_score > 0 AND confidence_score < 7 THEN 2
                    WHEN total_turns > 0 THEN 3
                    ELSE 4
                END ASC,
                last_studied_at DESC
            LIMIT 1
            """
            cursor.execute(query, tuple(params))
            row = cursor.fetchone()
            if not row:
                return None

            data = dict(row)
            concept_name = data["concept_name"]
            conf = data.get("confidence_score") or 0
            struggle = data.get("struggle_count") or 0

            return {
                "concept_id": data["concept_id"],
                "concept_name": concept_name,
                "category": data["category"],
                "last_studied_at": data["last_studied_at"],
                "confidence_score": conf,
                "struggle_count": struggle,
                "prompt_suggestion": (
                    f"Welcome back! Last time, you were working on **{concept_name}** "
                    f"({'Confidence: ' + str(conf) + '/10' if conf > 0 else 'under active practice'}). "
                    f"How are you feeling about {concept_name} today, or would you like a quick 1-minute recall check?"
                )
            }
