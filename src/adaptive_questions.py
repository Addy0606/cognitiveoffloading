"""
Adaptive Feedback-Driven Diagnostic & Challenge Question Engine.

When a student submits comprehension or confidence feedback for a concept:
- Tier 1: Low Confidence (Rating 1-4 or 'confused') -> Simple diagnostic MCQ with 4 options to identify foundational misconceptions without cognitive overload.
- Tier 2: Medium Confidence (Rating 5-7) -> Logic / Code tracing question to test practical mechanics.
- Tier 3: High Confidence (Rating 8-10 or 'clear') -> Advanced edge-case & computational complexity optimization challenge to verify deep mastery.
"""

from typing import Dict, Any, Optional

ADAPTIVE_QUESTION_BANK: Dict[str, Dict[str, Any]] = {
    "avl_trees": {
        "concept_name": "AVL Trees & Self-Balancing",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "To help clear up any confusion on AVL Trees, let's verify the core rule:\n\n"
                "**Question:** An AVL tree node is considered unbalanced if its balance factor (`height(left) - height(right)`) is:\n\n"
                "- **A)** Exactly 0\n"
                "- **B)** Either -1, 0, or +1\n"
                "- **C)** Greater than +1 or less than -1 (i.e. +2 or -2)\n"
                "- **D)** Any negative number\n\n"
                "*Which option (A, B, C, or D) do you think is correct? Reply with your choice!*"
            ),
            "expected_answer": "C"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Great! Let's test your ability to trace an AVL rebalance operation:\n\n"
                "**Scenario:** You insert nodes in the order `[30, 20, 10]` into an initially empty AVL tree.\n\n"
                "1. After inserting `10`, which node has a balance factor of `+2`?\n"
                "2. Is this an **LL, RR, LR, or RL** imbalance, and which single rotation restores balance?\n\n"
                "*Give it a try in your own words!*"
            ),
            "expected_answer": "LL imbalance at 30, resolved by a single Right Rotation around 30."
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Edge Cases & Invariants)**\n\n"
                "Since you feel confident with AVL Trees, here is an optimization challenge:\n\n"
                "**Question:** Why does an AVL tree guarantee $O(\\log N)$ worst-case search time while a standard BST can degrade to $O(N)$? "
                "Specifically, what is the maximum possible height of an AVL tree with $N$ nodes in terms of the Fibonacci sequence, "
                "and under what insertion pattern does a double rotation (LR or RL) become strictly necessary instead of a single rotation?"
            ),
            "expected_answer": "Height bounded by ~1.44 log2(N); double rotation required when the newly inserted node lies in the inner subtree."
        }
    },
    "linked_lists": {
        "concept_name": "Reverse a Linked List",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "Let's make sure the pointer mechanics are crystal clear:\n\n"
                "**Question:** When reversing a singly linked list iteratively, why do we need a temporary pointer (`curr.next`) *before* changing `curr.next = prev`?\n\n"
                "- **A)** To keep track of the head of the list\n"
                "- **B)** Because changing `curr.next` overwrites the forward link, losing the rest of the list\n"
                "- **C)** To check if the list contains cycles\n"
                "- **D)** To count the total length of the list\n\n"
                "*Which option (A, B, C, or D) is the reason?*"
            ),
            "expected_answer": "B"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Let's trace a 3-node list: `1 -> 2 -> 3 -> None`.\n\n"
                "At the start of the first iteration (`curr = 1`, `prev = None`):\n"
                "1. What is `next_temp`?\n"
                "2. What does `1.next` point to after the reversal step?\n"
                "3. What are `prev` and `curr` at the start of the second iteration?"
            ),
            "expected_answer": "next_temp=2, 1.next=None, prev=1, curr=2"
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Edge Cases & Invariants)**\n\n"
                "You rated high confidence! Let's test your depth:\n\n"
                "**Question:** How would you reverse a linked list **in groups of $K$ nodes** (LeetCode 25 - Reverse Nodes in k-Group) in $O(N)$ time and $O(1)$ extra space? "
                "What is the edge case if the remaining nodes at the end of the list are fewer than $K$?"
            ),
            "expected_answer": "Leave remaining nodes unmodified; reconnect sub-lists using pointer tracking."
        }
    },
    "arrays_strings": {
        "concept_name": "Arrays & Two-Pointer Technique",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** In the Two-Pointer technique for a sorted array (e.g. Two Sum II), if `arr[left] + arr[right] < target`, what must we do next?\n\n"
                "- **A)** Increment `left` (`left += 1`)\n"
                "- **B)** Decrement `right` (`right -= 1`)\n"
                "- **C)** Reset both pointers to 0\n"
                "- **D)** Swap `arr[left]` and `arr[right]`\n\n"
                "*Which option (A, B, C, or D) increases the sum?*"
            ),
            "expected_answer": "A"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Suppose you have `arr = [1, 2, 4, 7, 11, 15]` and `target = 15`.\n"
                "Trace each step: what are `left`, `right`, the current sum, and which pointer moves at each turn?"
            ),
            "expected_answer": "left=0, right=5 (sum 16 > 15 -> right=4); left=0, right=4 (sum 12 < 15 -> left=1); left=1, right=4 (sum 13 < 15 -> left=2); left=2, right=4 (4+11=15 -> found!)."
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Edge Cases & Invariants)**\n\n"
                "**Question:** Explain how the 3Sum problem ($A + B + C = 0$) handles duplicate triplets in $O(N^2)$ time without using extra space (like a Hash Set). What pointer checks prevent duplicate combinations?"
            ),
            "expected_answer": "Sort first, skip identical adjacent elements using while loops on left and right."
        }
    },
    "binary_trees": {
        "concept_name": "Binary Trees & BSTs",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** Which tree traversal visits nodes in non-decreasing sorted order for a valid Binary Search Tree (BST)?\n\n"
                "- **A)** Pre-order (Root, Left, Right)\n"
                "- **B)** In-order (Left, Root, Right)\n"
                "- **C)** Post-order (Left, Right, Root)\n"
                "- **D)** Level-order (BFS)\n\n"
                "*Which option (A, B, C, or D) produces sorted output?*"
            ),
            "expected_answer": "B"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Given this BST:\n"
                "```\n"
                "       8\n"
                "      / \\\n"
                "     3   10\n"
                "    / \\\n"
                "   1   6\n"
                "```\n"
                "Write down the exact sequence of nodes printed by an **In-order Traversal**."
            ),
            "expected_answer": "[1, 3, 6, 8, 10]"
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Edge Cases & Invariants)**\n\n"
                "**Question:** When validating whether a binary tree is a valid BST, why is checking `node.left.val < node.val < node.right.val` locally insufficient? "
                "How do you formulate the recursive invariant with `(-infinity, +infinity)` bounds?"
            ),
            "expected_answer": "Subtree nodes must satisfy global ancestor bounds, not just direct parent."
        }
    },
    "dynamic_programming": {
        "concept_name": "Dynamic Programming & Memoization",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** What are the two essential properties a problem must possess for Dynamic Programming to apply?\n\n"
                "- **A)** Sorted inputs and Two Pointers\n"
                "- **B)** Overlapping Subproblems and Optimal Substructure\n"
                "- **C)** Linear Time and Constant Space\n"
                "- **D)** Greediness and Binary Choices\n\n"
                "*Which option (A, B, C, or D) describes DP?*"
            ),
            "expected_answer": "B"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "In the 0/1 Knapsack problem with weights `[2, 3]`, values `[3, 4]`, and capacity `W = 5`:\n"
                "What is the state transition recurrence for `dp[i][w]`, and what value is stored in `dp[2][5]`?"
            ),
            "expected_answer": "dp[i][w] = max(dp[i-1][w], val[i] + dp[i-1][w-wt[i]]); dp[2][5] = 7"
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Edge Cases & Invariants)**\n\n"
                "**Question:** In the Coin Change problem (fewest coins to make amount $V$), explain how you can optimize the standard 2D DP table into a 1D array. "
                "Why must the 1D array iterate forwards for unbounded knapsack (unlimited coins) vs backwards for 0/1 knapsack (each item once)?"
            ),
            "expected_answer": "Forward iteration re-uses newly computed values from current item; backward iteration ensures items only used once."
        }
    },
    "graph_algorithms": {
        "concept_name": "Graph BFS & DFS",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** Which data structure is fundamentally used to implement Breadth-First Search (BFS)?\n\n"
                "- **A)** Stack (LIFO)\n"
                "- **B)** Queue (FIFO)\n"
                "- **C)** Priority Queue / Min-Heap\n"
                "- **D)** Disjoint Set Union (DSU)\n\n"
                "*Which option (A, B, C, or D) guarantees level-by-level traversal?*"
            ),
            "expected_answer": "B"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Why does BFS find the **shortest path** in an unweighted graph, but fails to guarantee the shortest path in a graph with positive weighted edges? What algorithm should be used instead?"
            ),
            "expected_answer": "BFS expands uniformly by edge count; for positive weighted edges, Dijkstra's algorithm with a priority queue is required."
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Edge Cases & Invariants)**\n\n"
                "**Question:** How does Tarjan's or Kosaraju's algorithm identify Strongly Connected Components (SCCs) in a directed graph in $O(V + E)$? Explain the role of discovery times (`tin`) and low-link values (`low`)."
            ),
            "expected_answer": "Low-link value identifies the highest ancestor reachable; when low[u] == tin[u], u is the root of an SCC."
        }
    }
}

def get_adaptive_intervention(
    concept_id: str,
    rating: Optional[int] = None,
    label: Optional[str] = None,
    concept_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a tailored follow-up question / diagnostic check based on student feedback rating.
    """
    cid = concept_id.lower().replace("-", "_").strip() if concept_id else ""
    bank_entry = ADAPTIVE_QUESTION_BANK.get(cid)

    # Determine Tier
    is_low = (rating is not None and rating <= 4) or label == "confused"
    is_high = (rating is not None and rating >= 8) or label == "clear"

    tier_key = "low" if is_low else ("high" if is_high else "medium")
    c_name = concept_name or (bank_entry["concept_name"] if bank_entry else cid.replace("_", " ").title())

    if bank_entry and tier_key in bank_entry:
        item = bank_entry[tier_key]
        return {
            "concept_id": cid,
            "concept_name": c_name,
            "tier": item["tier"],
            "prompt": item["prompt"],
            "expected_answer": item.get("expected_answer", "")
        }

    # Fallback generic templates if concept is not explicitly in bank
    if is_low:
        return {
            "concept_id": cid,
            "concept_name": c_name,
            "tier": "Tier 1: Foundational Diagnostic Check",
            "prompt": (
                f"💡 **Diagnostic Check for {c_name}**\n\n"
                f"You noted some confusion on **{c_name}**. Let's break it down to basics:\n\n"
                f"What is the core purpose of **{c_name}**, and what is the biggest challenge or stumbling block you ran into? "
                "Tell me in a short sentence, and I will walk you through a simple step-by-step example!"
            )
        }
    elif is_high:
        return {
            "concept_id": cid,
            "concept_name": c_name,
            "tier": "Tier 3: Advanced Mastery Challenge",
            "prompt": (
                f"🚀 **Mastery Challenge for {c_name}**\n\n"
                f"Awesome confidence on **{c_name}**! Let's test your depth:\n\n"
                f"What is the worst-case time and space complexity of operations in **{c_name}**, "
                "and what specific input or edge-case triggers that worst-case performance?"
            )
        }
    else:
        return {
            "concept_id": cid,
            "concept_name": c_name,
            "tier": "Tier 2: Application Practice",
            "prompt": (
                f"🎯 **Progress Check for {c_name}**\n\n"
                f"You're building good familiarity with **{c_name}**. "
                "Can you describe one concrete scenario or problem where you would choose this technique over a naive brute-force approach?"
            )
        }
