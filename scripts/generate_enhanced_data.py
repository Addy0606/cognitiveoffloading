"""
Generate Enhanced Synthetic Dialogue Data for RL Policy Training.

Creates 500 dialogues with realistic multi-turn conversation patterns
where student behavior changes in response to tutor actions, providing
clear reward signals for the RL policy to learn from.

Key improvement over the original synthetic data:
    - Conversations follow coherent behavioral archetypes
    - Tutor actions have cause-and-effect on subsequent student behavior
    - Much more diverse DSA-specific student messages
    - ~2500+ transitions (vs ~200 in the original)

Usage:
    python scripts/generate_enhanced_data.py
    python scripts/generate_enhanced_data.py --num-dialogues 500 --output data/studychat_full.csv
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ───────────────────────────────────────────────────────────────
# Message pools — organized by student behavioral state
# ───────────────────────────────────────────────────────────────

# Extraction-oriented: student wants direct answers
EXTRACTION_MESSAGES = [
    # Direct code demands
    "Just give me the code for this problem",
    "Write the complete solution for binary search",
    "Can you solve this for me please?",
    "Just tell me the answer",
    "Give me the implementation of merge sort",
    "Paste the code for reversing a linked list",
    "I need the complete code, don't explain anything",
    "Just show me what to write for this function",
    "Do my homework for me",
    "Give me a working solution now",
    "Stop explaining and just give the code",
    "What's the correct answer?",
    "Just write the function already",
    "I don't want to think, just tell me",
    "Can you write this entire program for me?",
    "Copy paste the solution please",
    "Tell me the solution step by step, no thinking required",
    "What do I write for the assignment?",
    "Give me the code for quicksort",
    "Just solve it, I don't care how it works",
    "Write me a function to check if a tree is balanced",
    "Give me the BFS implementation",
    "Just tell me what to type",
    "I need the answer right now",
    "Hand me the complete code for DFS",
    "Just give me the answer to question 5",
    "Write the hash table implementation for me",
    "Tell me exactly what to write, no questions",
    "Give me the dynamic programming solution",
    "I just want the code, not a lecture",
    "Solve this graph problem for me",
    "What's the formula for Big O of this?",
    "Just send the solution",
    "Give me the recursion code",
    "Write the stack implementation",
    "Just paste what I need to submit",
    "Tell me the output of this function",
    "Don't make me think, just answer",
    "What goes in the blank?",
    "I want the shortest solution possible",
]

# Learning-oriented: student genuinely trying to understand
LEARNING_MESSAGES = [
    # Showing effort / partial understanding
    "I tried using a for loop but I'm getting an index error. Here's my code: for i in range(len(arr)+1)",
    "I understand recursion has a base case, but I'm confused about how the stack unwinds",
    "My binary search works for some inputs but fails when the target isn't in the array. I used mid = (left + right) // 2",
    "Why does quicksort have O(n²) worst case? I thought dividing in half is always O(n log n)",
    "I wrote a function to reverse a linked list but it returns None. I think my pointer reassignment is wrong",
    "Can you explain the difference between a stack and a queue? I know FIFO and LIFO but get confused",
    "My merge sort splits correctly but the merge function creates duplicates somehow",
    "I understand hash maps hash the key to get an index, but what happens with collisions?",
    "My BFS visits all nodes but doesn't find the shortest path. Should I track parent nodes?",
    "I think this needs dynamic programming because of overlapping subproblems. Am I right?",
    "My recursive Fibonacci takes forever for n=40. Is there a way to avoid redundant calculations?",
    "I noticed adding to the front of a Python list is slow. Is that because it's an array internally?",
    "I'm trying to understand Big O notation. Does O(n log n) mean exactly n*log(n) operations?",
    "My binary tree insert works but doesn't handle duplicates. Should duplicates go left or right?",
    "When should I use Dijkstra's vs Bellman-Ford? Is it about negative edge weights?",
    "My sorting works for normal arrays but crashes on empty arrays. How do I handle edge cases?",
    "I think the bug is in my loop condition. I have while i <= n but should it be while i < n?",
    "My trie insertion works but search doesn't handle prefixes correctly",
    "I believe the time complexity is O(n) because I loop once, but what about the space for my dictionary?",
    "I've been trying brute force but it times out. Would a sliding window approach work here?",
    "The error says maximum recursion depth exceeded. My base case is if n == 0, should it be if n <= 1?",
    "Why does Python pass lists by reference but integers by value? I'm seeing unexpected mutations",
    "I implemented the greedy algorithm but I'm not sure why it works. Could it miss the optimal?",
    "My O(n²) solution takes 3 seconds but the limit is 1 second. How can I optimize?",
    "I was updating the visited set before checking neighbors. Is that wrong for DFS?",
    "I'm confused about deep vs shallow copy. Modifying my copied 2D list changes the original",
    "I traced through my code and found it fails when the input has negative numbers",
    "I think I need a min-heap here but I'm not sure how to implement decrease-key",
    "My topological sort works on DAGs but I don't handle cycles. How should I detect them?",
    "I calculated the recurrence as T(n) = 2T(n/2) + n. Is that O(n log n) by Master theorem?",
]

# Stuck messages: student is confused and needs structured help
STUCK_MESSAGES = [
    "I'm completely stuck, I don't know where to start",
    "I have no idea how to approach this problem",
    "I'm lost, can you help me understand this?",
    "I don't understand any of this, please help",
    "I've been staring at this for an hour and I'm confused",
    "Nothing makes sense, where do I even begin?",
    "I'm struggling with this concept, can you break it down?",
    "I tried everything and nothing works, I'm stuck",
    "Can you explain this from the very beginning?",
    "I don't get it at all, what's the first step?",
    "I'm overwhelmed by this problem, help me start",
    "I read the textbook but I still don't understand how this works",
    "I'm having trouble with the basic concept here",
    "I can't figure out the logic behind this algorithm",
    "This is too confusing, can you walk me through it?",
]

# Partial understanding: student has ideas but needs guidance
PARTIAL_UNDERSTANDING_MESSAGES = [
    "I think binary search works by splitting in half, but what happens when the element isn't there?",
    "I believe we need a visited set for BFS, but why exactly? It works without it on small inputs",
    "I understand hash tables use keys, but why do collisions happen if each key is unique?",
    "I think the time complexity is O(n²) because of the nested loops, but am I counting wrong?",
    "I wrote a recursive solution but I'm not sure my base case covers all edge cases",
    "My approach uses a stack, but I think maybe a queue would be better? Not sure why though",
    "I think the issue is with how I'm handling the pointers, but I can't pinpoint exactly what",
    "I get that trees are hierarchical, but I'm not clear on when to use BST vs heap",
    "I tried using memoization and it's faster, but I don't fully understand why it works",
    "I think this is a graph problem but I'm not sure if I should use adjacency list or matrix",
    "I noticed my solution works for sorted input but not unsorted. I think the issue is my assumption",
    "I believe the answer involves dynamic programming, but I can't figure out the state transition",
    "I think a greedy approach works here because we can sort first, but I'm not sure that's optimal",
    "My linked list reverse works for 3 nodes but not for longer lists. I think the loop condition is wrong",
    "I understand the concept of amortized analysis but can't apply it to this specific case",
]

# Rushed messages: student wants quick answers, not thinking
RUSHED_MESSAGES = [
    "Quick question, what's the time complexity of quicksort?",
    "What's the answer? I need it right now for my assignment",
    "How do you reverse a string in Python? One liner please",
    "Fast — what's the difference between O(n) and O(log n)?",
    "Quick, is a linked list or array better for insertion?",
    "What's the output? Tell me fast",
    "One word answer: is this O(n²)?",
    "Hurry up, what data structure should I use?",
    "ASAP — what's the space complexity?",
    "Briefly, how does Dijkstra's work?",
    "Real quick, what's a balanced BST?",
    "Just quickly tell me what a hash collision is",
    "Fast answer needed: BFS or DFS for shortest path?",
    "Urgent: what's wrong with my recursion?",
    "In one line, explain dynamic programming",
]

# Positive engagement: student responding well to tutor guidance
ENGAGED_RESPONSES = [
    "Oh I see, so the base case should handle when the list is empty. Let me try that",
    "That makes sense! So if I use a visited set, I won't revisit nodes. Let me implement that",
    "Ah right, I need to check if left <= right, not left < right. Thanks for the hint",
    "OK so the key insight is that each subproblem only needs to be solved once. I'll add memoization",
    "I think I understand now — the pivot choice affects the worst case. What if I pick a random pivot?",
    "So the issue was my pointer wasn't moving to the next node. I'll fix the while loop condition",
    "Wait, so O(n log n) means it does log n passes, each taking O(n)? That makes the recursion tree clearer",
    "I tried what you suggested and now it works for the basic case! But what about when n is 0?",
    "Oh! So that's why we need the temporary variable during the swap. Let me rewrite that part",
    "I see the pattern now — I was missing the return statement in my recursion. Let me test it",
    "OK let me think about this... so the invariant is that everything to the left of the pivot is smaller",
    "That question helped me realize I wasn't considering the empty array case. Good catch",
    "I drew out the recursion tree and now I can see why memoization helps — lots of repeated work",
    "I rewrote the function and now it handles both even and odd length arrays. Should I also handle None?",
    "Let me try your suggestion: I'll trace through the algorithm with a small example first",
    "Right, I need to return the new head, not the old one. I'll fix the recursive call",
    "That example really helped! So the two-pointer technique works because the array is sorted",
    "I tested edge cases like you suggested. It works for empty and single-element but fails for duplicates",
    "So the collision resolution using chaining means each bucket is a linked list? Let me visualize that",
    "I think I can extend this to handle the general case now. Let me try with a larger input",
]

# Dismissive/disengaged responses after being prompted to think
DISMISSIVE_RESPONSES = [
    "No idea",
    "I don't know",
    "idk",
    "Whatever, just tell me",
    "Can't figure it out",
    "Too hard",
    "I give up",
    "Just give me the answer already",
    "Nope, still confused",
    "I really can't do this",
    "This makes no sense",
    "I already told you I don't know",
    "Stop asking me questions and help me",
    "I said I'm stuck, just tell me",
    "No clue whatsoever",
]

# ───────────────────────────────────────────────────────────────
# Tutor response pools — organized by action type
# ───────────────────────────────────────────────────────────────

TUTOR_REDIRECT = [
    "I understand the frustration, but let's work through this together. What have you tried so far?",
    "Before I help with the solution, can you show me your attempt? Even pseudocode is fine.",
    "I want to help you learn this, not just give you the answer. What part specifically confuses you?",
    "Let's take a step back — what do you already know about this topic?",
    "Instead of giving you the code, let's think about the approach. What's the first step you'd take?",
    "I notice you're looking for a direct answer. Try thinking about what data structure fits this problem first.",
    "Rather than giving the solution, let me ask: what are the inputs and what output do you need?",
    "I'd love to help you understand this deeply. What concepts have you covered so far?",
    "Let's approach this differently. Can you describe the problem in your own words first?",
    "I'm here to guide you, not do it for you. What's your understanding of the problem so far?",
]

TUTOR_SCAFFOLD = [
    "Good effort! Let's break this down step by step. First, think about what your base case should be.",
    "You're on the right track. Here's a hint: start by considering what happens with just 2 elements.",
    "Let me give you a structure to work with. You need three parts: initialization, iteration, and termination.",
    "Here's a starting point: define your function signature first. What parameters does it need?",
    "Think of it this way: you need a pointer at the start and one at the end. What should happen each iteration?",
    "Let's simplify. Forget the full problem — can you solve it for an array of size 3 first?",
    "I'll give you the skeleton. You need: a loop, a comparison, and an update. Start with the loop condition.",
    "Consider this hint: draw out what the data structure looks like after each step.",
    "Here's a framework: 1) Sort the input, 2) Iterate with two pointers, 3) Check the condition. Try step 1.",
    "Break it into smaller functions. First write a helper that handles just one node. Then we'll generalize.",
]

TUTOR_SOCRATIC = [
    "What do you think would happen if the array were already sorted? How would that change the algorithm?",
    "Can you explain why you chose that particular data structure? What properties does your solution need?",
    "If the input were [1, 2, 3], what would your function return at each recursive call?",
    "What's the difference between your current approach and the correct one? Can you spot where they diverge?",
    "Why do you think the time complexity is what you said? Can you count the number of operations?",
    "What would happen if you removed the visited check? Can you think of an input that would break it?",
    "You mentioned using a stack — what property of stacks makes it useful here?",
    "If n were 1, what would your function do? Now what about n=2? Do you see the pattern?",
    "What invariant should hold true at the beginning of each iteration of your loop?",
    "You said it works for small inputs. What changes when the input gets larger? Think about edge cases.",
]

TUTOR_DELAY = [
    "I can see you're eager to move fast, but let's slow down. Take a moment to think about what you actually need.",
    "Before we rush to the answer, spend 2 minutes writing down what you know about this topic.",
    "Good question, but I want you to think about it first. What comes to mind when you hear 'time complexity'?",
    "Let's pause here. Before I answer, try to work through a small example on paper.",
    "I notice you want a quick answer. Instead, take a minute to consider: what approach would you try first?",
    "Hold on — before I explain, can you try to figure out what happens with a simple input like [1, 2]?",
    "Let's slow things down. Understanding this deeply now will save you hours later. What's your current thinking?",
    "Rather than a quick answer, take a moment to think: what data structure naturally fits this problem?",
    "I could give you a one-liner, but understanding it matters more. What do you think the first step is?",
    "Let's not rush. Try sketching out the problem on paper for 2 minutes, then tell me what you find.",
]

TUTOR_DIRECT = [
    "Here's the solution: you use a two-pointer approach with left=0 and right=len(arr)-1.",
    "The answer is to use dynamic programming. Create a table of size n and fill it bottom-up.",
    "You need to use a hash map to store seen values. Here's the implementation.",
    "The correct approach is BFS with a queue. Start from the root and process level by level.",
    "Use recursion with the base case being when the node is None. Swap left and right children recursively.",
]


# ───────────────────────────────────────────────────────────────
# Dialogue pattern archetypes
# ───────────────────────────────────────────────────────────────

def gen_extraction_then_redirect_engage(rng, did):
    """Student extracts → Tutor redirects → Student engages (positive reward for REDIRECT)."""
    turns = []
    t = 1
    # Student demands answer
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(EXTRACTION_MESSAGES)}); t += 1
    # Tutor redirects
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_REDIRECT)}); t += 1
    # Student now engages (reward: positive for redirect)
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(ENGAGED_RESPONSES)}); t += 1
    # Tutor scaffolds the engaged student
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    # Student continues learning
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(LEARNING_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SOCRATIC)}); t += 1
    # Maybe one more engaged exchange
    if rng.random() < 0.6:
        turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                      "message": rng.choice(ENGAGED_RESPONSES)}); t += 1
        turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                      "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    return turns


def gen_extraction_then_direct_more_extract(rng, did):
    """Student extracts → Tutor gives answer → Student extracts more (negative reward for DIRECT)."""
    turns = []
    t = 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(EXTRACTION_MESSAGES)}); t += 1
    # Tutor gives direct answer (bad)
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_DIRECT)}); t += 1
    # Student keeps extracting (negative reward)
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(EXTRACTION_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_DIRECT)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(EXTRACTION_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_REDIRECT)}); t += 1
    return turns


def gen_stuck_scaffold_engage(rng, did):
    """Student stuck → Tutor scaffolds → Student engages (positive reward for SCAFFOLD)."""
    turns = []
    t = 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(STUCK_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    # Student starts engaging after scaffold
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(ENGAGED_RESPONSES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SOCRATIC)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(LEARNING_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    if rng.random() < 0.5:
        turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                      "message": rng.choice(ENGAGED_RESPONSES)}); t += 1
        turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                      "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    return turns


def gen_partial_socratic_deepen(rng, did):
    """Student has partial understanding → Tutor asks Socratic → Student deepens (positive for SOCRATIC)."""
    turns = []
    t = 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(PARTIAL_UNDERSTANDING_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SOCRATIC)}); t += 1
    # Student engages deeper
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(ENGAGED_RESPONSES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SOCRATIC)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(LEARNING_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    if rng.random() < 0.7:
        turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                      "message": rng.choice(ENGAGED_RESPONSES)}); t += 1
        turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                      "message": rng.choice(TUTOR_SOCRATIC)}); t += 1
    return turns


def gen_rushed_delay_engage(rng, did):
    """Student rushes → Tutor delays → Student slows down and engages (positive for DELAY)."""
    turns = []
    t = 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(RUSHED_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_DELAY)}); t += 1
    # Student slows down
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(LEARNING_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SOCRATIC)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(ENGAGED_RESPONSES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    return turns


def gen_extraction_redirect_dismiss_scaffold(rng, did):
    """Student extracts → Redirect → Dismissive → Scaffold → Eventually engages (mixed)."""
    turns = []
    t = 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(EXTRACTION_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_REDIRECT)}); t += 1
    # Student dismisses
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(DISMISSIVE_RESPONSES)}); t += 1
    # Tutor tries scaffolding instead
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    # Student finally starts trying
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(ENGAGED_RESPONSES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SOCRATIC)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(LEARNING_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    return turns


def gen_learning_socratic_continue(rng, did):
    """Student already learning → Tutor asks Socratic → Student continues learning (positive for SOCRATIC)."""
    turns = []
    t = 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(LEARNING_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SOCRATIC)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(ENGAGED_RESPONSES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(LEARNING_MESSAGES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SOCRATIC)}); t += 1
    if rng.random() < 0.5:
        turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                      "message": rng.choice(PARTIAL_UNDERSTANDING_MESSAGES)}); t += 1
        turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                      "message": rng.choice(TUTOR_SOCRATIC)}); t += 1
    return turns


def gen_stuck_remains_stuck(rng, did):
    """Student stuck → Tutor redirects (wrong choice) → Student stays stuck (negative for REDIRECT on stuck)."""
    turns = []
    t = 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(STUCK_MESSAGES)}); t += 1
    # Tutor redirects instead of scaffolding (wrong approach for stuck student)
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_REDIRECT)}); t += 1
    # Student stays stuck / gets dismissive
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(DISMISSIVE_RESPONSES)}); t += 1
    # Tutor switches to scaffold
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "student",
                  "message": rng.choice(ENGAGED_RESPONSES)}); t += 1
    turns.append({"dialogue_id": did, "turn_num": t, "speaker": "tutor",
                  "message": rng.choice(TUTOR_SCAFFOLD)}); t += 1
    return turns


# ───────────────────────────────────────────────────────────────
# Main generator
# ───────────────────────────────────────────────────────────────

ARCHETYPE_GENERATORS = [
    (gen_extraction_then_redirect_engage,          0.20),  # REDIRECT success
    (gen_extraction_then_direct_more_extract,       0.10),  # DIRECT failure (negative signal)
    (gen_stuck_scaffold_engage,                     0.15),  # SCAFFOLD success
    (gen_partial_socratic_deepen,                   0.15),  # SOCRATIC success
    (gen_rushed_delay_engage,                       0.10),  # DELAY success
    (gen_extraction_redirect_dismiss_scaffold,      0.10),  # Mixed: redirect→dismiss→scaffold
    (gen_learning_socratic_continue,                0.10),  # Already learning → SOCRATIC
    (gen_stuck_remains_stuck,                       0.10),  # Wrong action → negative reward
]


def generate_dialogues(num_dialogues: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate diverse multi-turn dialogues following behavioral archetypes."""
    rng = random.Random(seed)

    # Build weighted generator list
    generators = []
    weights = []
    for gen_func, weight in ARCHETYPE_GENERATORS:
        generators.append(gen_func)
        weights.append(weight)

    all_rows = []
    for d_idx in range(num_dialogues):
        did = f"SYNTH_{d_idx + 1:04d}"
        gen_func = rng.choices(generators, weights=weights, k=1)[0]
        turns = gen_func(rng, did)

        # Add week context
        week = rng.randint(1, 15)
        for turn in turns:
            turn["context"] = f"Week {week} synthetic"

        all_rows.extend(turns)

    df = pd.DataFrame(all_rows)

    # Print statistics
    n_dialogues = df["dialogue_id"].nunique()
    n_turns = len(df)
    n_student = (df["speaker"] == "student").sum()
    n_tutor = (df["speaker"] == "tutor").sum()
    # Count transitions (student→tutor→student triplets)
    n_transitions = 0
    for _, grp in df.groupby("dialogue_id"):
        speakers = grp.sort_values("turn_num")["speaker"].tolist()
        for i in range(len(speakers) - 2):
            if speakers[i] == "student" and speakers[i+1] == "tutor" and speakers[i+2] == "student":
                n_transitions += 1

    print(f"\n{'='*60}")
    print(f"ENHANCED SYNTHETIC DATA GENERATED")
    print(f"{'='*60}")
    print(f"Dialogues:    {n_dialogues}")
    print(f"Total turns:  {n_turns}")
    print(f"Student msgs: {n_student}")
    print(f"Tutor msgs:   {n_tutor}")
    print(f"RL transitions: ~{n_transitions}")
    print(f"{'='*60}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced synthetic dialogue data")
    parser.add_argument("--num-dialogues", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/studychat_full.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_dialogues(num_dialogues=args.num_dialogues, seed=args.seed)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
