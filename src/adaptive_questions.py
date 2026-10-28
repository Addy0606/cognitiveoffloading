"""
Adaptive Feedback-Driven Diagnostic & Challenge Question Engine.

When a student submits comprehension or confidence feedback for a concept:
- Tier 1: Low Confidence (Rating 1-4 or 'confused') -> Simple diagnostic MCQ with 4 options to identify foundational misconceptions without cognitive overload.
- Tier 2: Medium Confidence (Rating 5-7) -> Logic / Code tracing question to test practical mechanics.
- Tier 3: High Confidence (Rating 8-10 or 'clear') -> Advanced edge-case & computational complexity optimization challenge to verify deep mastery.
"""

from typing import Dict, Any, Optional

ADAPTIVE_QUESTION_BANK: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------
    # 1. AVL Trees & Self-Balancing
    # -------------------------------------------------------------
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
                "- **C)** Greater than +1 or less than -1 (e.g. +2 or -2)\n"
                "- **D)** Any negative number\n\n"
                "*Which option (A, B, C, or D) is correct? Reply with your choice!*"
            ),
            "expected_answer": "C"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Let's test your ability to trace an AVL rebalance operation:\n\n"
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
                "Specifically, what is the maximum height bound in terms of the golden ratio / Fibonacci sequence, "
                "and under what insertion pattern does a double rotation (LR or RL) become strictly necessary instead of a single rotation?"
            ),
            "expected_answer": "Height bounded by ~1.44 log2(N); double rotation required when the newly inserted node lies in the inner subtree."
        }
    },

    # -------------------------------------------------------------
    # 2. Reverse a Linked List
    # -------------------------------------------------------------
    "reverse_linked_list": {
        "concept_name": "Reverse a Linked List",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "Let's make sure the pointer mechanics are crystal clear:\n\n"
                "**Question:** When reversing a singly linked list iteratively, why do we store `next_node = curr.next` *before* changing `curr.next = prev`?\n\n"
                "- **A)** To keep track of the head of the list\n"
                "- **B)** Because changing `curr.next` breaks the forward link, losing the rest of the list\n"
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
    "linked_lists": {
        "concept_name": "Reverse a Linked List",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** When reversing a singly linked list iteratively, why do we need `next_node = curr.next` before changing `curr.next = prev`?\n\n"
                "- **A)** To verify head node\n"
                "- **B)** Overwriting `curr.next` cuts the link to remaining nodes\n"
                "- **C)** To detect cycles\n"
                "- **D)** None of the above\n\n"
                "*Which option is correct?*"
            ),
            "expected_answer": "B"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": "🎯 **Progress Check (Tracing)**: Given `A -> B -> C`, trace the state of `prev` and `curr` after each iteration.",
            "expected_answer": "prev points to C, curr is None."
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery Challenge",
            "prompt": "🚀 **Mastery Challenge**: Explain recursive vs iterative linked list reversal time/space tradeoffs.",
            "expected_answer": "Iterative is O(1) space; recursive is O(N) auxiliary call stack space."
        }
    },

    # -------------------------------------------------------------
    # 3. Linked List Cycle Detection
    # -------------------------------------------------------------
    "cycle_detection": {
        "concept_name": "Linked List Cycle Detection",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** In Floyd's Tortoise and Hare algorithm, how many steps does `slow` and `fast` move per iteration?\n\n"
                "- **A)** slow moves 1 step, fast moves 1 step\n"
                "- **B)** slow moves 1 step, fast moves 2 steps\n"
                "- **C)** slow moves 2 steps, fast moves 3 steps\n"
                "- **D)** slow moves 0 steps, fast moves 2 steps\n\n"
                "*Which option (A, B, C, or D) is correct?*"
            ),
            "expected_answer": "B"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Why is Floyd's cycle detection guaranteed to run in $O(N)$ time instead of looping forever? "
                "How does the gap (relative distance) between `fast` and `slow` change on each step inside the loop?"
            ),
            "expected_answer": "The distance between fast and slow decreases by exactly 1 step on every iteration modulo the loop length."
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Cycle Entrance Proof)**\n\n"
                "**Question:** After `slow` and `fast` meet inside the cycle, Floyd's algorithm resets `slow` to the `head` "
                "and moves both `slow` and `fast` 1 step at a time until they collide again. "
                "Mathematically prove why this second collision is guaranteed to occur at the **exact entrance node** of the cycle!"
            ),
            "expected_answer": "Let distance to cycle start be L, distance from start to meeting point be d, cycle length be C. 2(L + d) = L + d + kC => L = kC - d."
        }
    },

    # -------------------------------------------------------------
    # 4. Binary Search
    # -------------------------------------------------------------
    "binary_search": {
        "concept_name": "Binary Search",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** Why do experienced programmers write `mid = low + (high - low) // 2` instead of `mid = (low + high) // 2`?\n\n"
                "- **A)** It executes faster in hardware\n"
                "- **B)** It avoids 32-bit integer overflow when `low + high` exceeds $2^{31} - 1$\n"
                "- **C)** It handles negative numbers differently\n"
                "- **D)** It works on unsorted arrays\n\n"
                "*Which option (A, B, C, or D) is the main reason?*"
            ),
            "expected_answer": "B"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Suppose you want to find the **first (leftmost) occurrence** of target `2` in `arr = [1, 2, 2, 2, 3, 5]`.\n\n"
                "When `arr[mid] == target`, what should you do with your pointers (`low` or `high`) so you don't stop prematurely at the middle 2?\n\n"
                "*Give the pointer update rule in one sentence!*"
            ),
            "expected_answer": "Set high = mid - 1 (or high = mid) while saving ans = mid, searching the left half."
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Search on Answer Space)**\n\n"
                "**Question:** In problems like *Koko Eating Bananas* or *Capacity to Ship Packages within D Days*, "
                "how do we apply Binary Search even when the input array is completely unsorted? "
                "What mathematical property (monotonic predicate $P(x)$) must the answer space satisfy?"
            ),
            "expected_answer": "The feasibility function must be monotonic (e.g. FFF...TTT); binary search finds the first True value."
        }
    },

    # -------------------------------------------------------------
    # 5. Binary Search Trees (BST)
    # -------------------------------------------------------------
    "binary_search_tree": {
        "concept_name": "Binary Search Trees (BST)",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** What fundamental invariant must hold true for EVERY node in a valid Binary Search Tree (BST)?\n\n"
                "- **A)** All nodes in the left subtree must be strictly less than the node, and all nodes in the right subtree must be strictly greater\n"
                "- **B)** The left child must simply be smaller than its immediate parent; descendants don't matter\n"
                "- **C)** The tree must always be completely balanced with equal depth\n"
                "- **D)** Every node must have exactly 2 children\n\n"
                "*Which option (A, B, C, or D) is the correct invariant?*"
            ),
            "expected_answer": "A"
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
                "1. What is the **In-order traversal** of this tree?\n"
                "2. If you delete node `3` (which has two children `1` and `6`), what node from the tree replaces it?"
            ),
            "expected_answer": "In-order: [1, 3, 6, 8, 10]. Node 3 is replaced by in-order successor 6 (or predecessor 1)."
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Validation Edge Cases)**\n\n"
                "**Question:** Consider the tree: Root=10, Left=5, Right=15, and 15's Left child is 6.\n"
                "Locally, `6 < 15` is valid, but the entire tree is NOT a valid BST. "
                "How does the optimal $O(N)$ BST validation algorithm propagate `(min_allowed, max_allowed)` bounds down the recursion to catch this violation?"
            ),
            "expected_answer": "Pass (low, high) ranges down recursively; for node.left pass (low, node.val), for node.right pass (node.val, high). 6 violates 10 < val < inf."
        }
    },
    "binary_trees": {
        "concept_name": "Binary Search Trees (BST)",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** Which traversal visits nodes in ascending sorted order for a valid BST?\n\n"
                "- **A)** Pre-order\n"
                "- **B)** In-order (Left, Root, Right)\n"
                "- **C)** Post-order\n"
                "- **D)** Level-order\n\n"
                "*Which option is correct?*"
            ),
            "expected_answer": "B"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": "🎯 **Progress Check**: Trace in-order traversal on a BST with nodes 5, 2, 7.",
            "expected_answer": "[2, 5, 7]"
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery Challenge",
            "prompt": "🚀 **Mastery Challenge**: Explain why degenerate BST degrades to O(N) and how balancing fixes it.",
            "expected_answer": "Skewed BST acts like linked list; self-balancing bounds height to O(log N)."
        }
    },

    # -------------------------------------------------------------
    # 6. Two Pointers Technique
    # -------------------------------------------------------------
    "two_pointers": {
        "concept_name": "Two Pointers Technique",
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
                "Suppose you have sorted `arr = [1, 2, 4, 7, 11, 15]` and `target = 15`.\n"
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
    "arrays_strings": {
        "concept_name": "Two Pointers Technique",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": "💡 **Diagnostic Check**: What pointer move increases the sum in a sorted two-pointer search? (Increment left).",
            "expected_answer": "Increment left"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": "🎯 **Progress Check**: Trace Two Sum II on [2, 7, 11, 15] with target 9.",
            "expected_answer": "left=0, right=1 (2+7=9 -> target found immediately)."
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery Challenge",
            "prompt": "🚀 **Mastery Challenge**: Explain Trapping Rain Water using two pointers in O(1) auxiliary space.",
            "expected_answer": "Maintain left_max and right_max, advance pointer with smaller max."
        }
    },

    # -------------------------------------------------------------
    # 7. 0/1 Knapsack Problem
    # -------------------------------------------------------------
    "knapsack_dp": {
        "concept_name": "0/1 Knapsack Problem",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** In the 0/1 Knapsack problem, why does a greedy approach (taking highest value-to-weight ratio first) FAIL to give the optimal answer?\n\n"
                "- **A)** Because weights are negative\n"
                "- **B)** Because items cannot be divided (fractional); taking an item may leave unused capacity that prevents fitting a better combination\n"
                "- **C)** Because greedy algorithms always take $O(2^N)$ time\n"
                "- **D)** Greedy actually always works for 0/1 Knapsack\n\n"
                "*Which option (A, B, C, or D) is correct?*"
            ),
            "expected_answer": "B"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Suppose capacity $W = 4$, and you have an item with `weight = 2`, `value = 10`.\n"
                "If `dp[w]` is the maximum value for capacity $w$:\n"
                "What is the recurrence relation to update `dp[w]` using this item?"
            ),
            "expected_answer": "dp[w] = max(dp[w], value + dp[w - weight])"
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Space Optimization)**\n\n"
                "**Question:** When optimizing the 0/1 Knapsack from a 2D table `dp[N+1][W+1]` to a 1D array `dp[W+1]`, "
                "why MUST the inner capacity loop iterate **backwards** (`for w in range(W, weight - 1, -1)`)? "
                "What bug happens if you iterate forwards?"
            ),
            "expected_answer": "Iterating forwards allows the same item to be reused multiple times (Unbounded Knapsack bug); backwards iteration ensures each item is used at most once."
        }
    },

    # -------------------------------------------------------------
    # 8. Coin Change (Min Coins)
    # -------------------------------------------------------------
    "coin_change": {
        "concept_name": "Coin Change (Min Coins)",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** In the Coin Change problem (finding minimum coins to make amount $A$), what is the base case `dp[0]`?\n\n"
                "- **A)** `dp[0] = 0` (0 coins needed to make amount 0)\n"
                "- **B)** `dp[0] = infinity`\n"
                "- **C)** `dp[0] = -1`\n"
                "- **D)** `dp[0] = 1`\n\n"
                "*Which option (A, B, C, or D) is the correct base case?*"
            ),
            "expected_answer": "A"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Given coins `[1, 2, 5]` and target `amount = 5`.\n"
                "Calculate `dp[1]`, `dp[2]`, `dp[3]`, `dp[4]`, and `dp[5]`."
            ),
            "expected_answer": "dp[1]=1, dp[2]=1, dp[3]=2, dp[4]=2, dp[5]=1"
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Unbounded vs 0/1)**\n\n"
                "**Question:** Contrast Coin Change (minimum coins, unlimited supply) with 0/1 Knapsack. "
                "In 1D DP array, which one iterates forward and which iterates backward? Why?"
            ),
            "expected_answer": "Coin change iterates forward because each coin can be used multiple times; 0/1 knapsack iterates backward to prevent reuse."
        }
    },

    # -------------------------------------------------------------
    # 9. Graph BFS (Shortest Path)
    # -------------------------------------------------------------
    "graph_bfs": {
        "concept_name": "Shortest Path (Graph BFS)",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** Which data structure is used to implement Breadth-First Search (BFS)?\n\n"
                "- **A)** Stack (LIFO)\n"
                "- **B)** Queue (FIFO)\n"
                "- **C)** Priority Queue (Min-Heap)\n"
                "- **D)** Binary Search Tree\n\n"
                "*Which option (A, B, C, or D) explores level by level?*"
            ),
            "expected_answer": "B"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Why must a node be marked as `visited` at the moment it is **pushed into the queue**, "
                "rather than when it is popped from the queue?"
            ),
            "expected_answer": "To prevent duplicate additions of the same node by multiple neighbors, which would cause exponential queue growth."
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Multi-Source BFS)**\n\n"
                "**Question:** In problems like *Rotting Oranges* (LeetCode 994) or finding the distance of the nearest 0 in a binary matrix, "
                "how does Multi-Source BFS guarantee optimal $O(R \\times C)$ time without running separate BFS passes for each source?"
            ),
            "expected_answer": "Push all initial sources into the queue simultaneously at time t=0; the single BFS wavefront propagates uniformly."
        }
    },

    # -------------------------------------------------------------
    # 10. Dijkstra's Shortest Path
    # -------------------------------------------------------------
    "dijkstra": {
        "concept_name": "Dijkstra's Shortest Path",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** Under what graph condition does Dijkstra's algorithm FAIL to find the correct shortest path?\n\n"
                "- **A)** Graphs with cycles\n"
                "- **B)** Graphs with directed edges\n"
                "- **C)** Graphs with negative edge weights\n"
                "- **D)** Graphs with more than 1000 nodes\n\n"
                "*Which option (A, B, C, or D) breaks Dijkstra's greedy invariant?*"
            ),
            "expected_answer": "C"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "What is the relaxation step in Dijkstra's algorithm when examining edge `(u, v)` with weight `w`? "
                "Under what condition do you update `dist[v]` and push `(dist[v], v)` into the min-heap?"
            ),
            "expected_answer": "If dist[u] + w < dist[v], set dist[v] = dist[u] + w and push (dist[v], v) to heap."
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Complexity & Alternatives)**\n\n"
                "**Question:** What is the precise time complexity of Dijkstra using a binary min-heap: $O((V + E) \\log V)$ or $O(E \\log V)$? "
                "If the graph contains negative edges but NO negative cycles, which algorithm must you use instead, and what is its complexity?"
            ),
            "expected_answer": "O((V + E) log V); use Bellman-Ford algorithm with O(V * E) complexity."
        }
    },

    # -------------------------------------------------------------
    # 11. Valid Parentheses (Stack)
    # -------------------------------------------------------------
    "valid_parentheses": {
        "concept_name": "Valid Parentheses (Stack)",
        "low": {
            "tier": "Tier 1: Foundational Diagnostic MCQ",
            "prompt": (
                "💡 **Diagnostic Check (Foundational Intuition)**\n\n"
                "**Question:** In the Valid Parentheses problem, when encountering a closing bracket (e.g. `')'`), what two checks must pass for the string to remain valid?\n\n"
                "- **A)** The stack must not be empty, AND the top element must be the matching opening bracket (`'('`)\n"
                "- **B)** The stack must be empty\n"
                "- **C)** The next character must also be a closing bracket\n"
                "- **D)** The string length must be odd\n\n"
                "*Which option (A, B, C, or D) is correct?*"
            ),
            "expected_answer": "A"
        },
        "medium": {
            "tier": "Tier 2: Application & Tracing",
            "prompt": (
                "🎯 **Progress Check (Logic Tracing)**\n\n"
                "Trace the stack contents character-by-character for `s = \"{[()]}\"`.\n"
                "At which step is the stack at its maximum depth, and what are its contents?"
            ),
            "expected_answer": "Max depth is 3 after pushing '(', contents: ['{', '[', '(']."
        },
        "high": {
            "tier": "Tier 3: Advanced Mastery & Edge-Case Challenge",
            "prompt": (
                "🚀 **Mastery Challenge (Constant Space Solution)**\n\n"
                "**Question:** For the *Longest Valid Parentheses* problem (LeetCode 32), a stack solution uses $O(N)$ space. "
                "How can you solve it in $O(1)$ auxiliary space using two linear passes (left-to-right and right-to-left) with `left` and `right` counter variables?"
            ),
            "expected_answer": "Traverse left-to-right: increment left for '(', right for ')'. If right > left, reset. If left == right, update max. Repeat right-to-left to catch unbalanced '('."
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
