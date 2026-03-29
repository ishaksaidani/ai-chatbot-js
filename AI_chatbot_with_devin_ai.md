# DEVIN AI TASK — Self-Improving AI Chatbot System

## Objective
Build a local AI chatbot system that:
- Learns from interactions over time
- Improves its responses incrementally
- Stores memory and context
- Can reason about user inputs
- Runs continuously and evolves (within safe limits)

IMPORTANT:
Do NOT attempt to train a large LLM from scratch. Instead, build a system that combines:
- Lightweight models (TensorFlow.js or Brain.js)
- Memory system
- Feedback loop
- Tool-based reasoning

---

## Core Architecture

The system must be modular and include:

### 1. Chat Engine
- Input → process → output pipeline
- Accept user messages
- Return structured responses

### 2. Memory System
- Store:
  - past conversations
  - user preferences
  - important facts
- Use JSON or SQLite
- Implement:
  - short-term memory (recent messages)
  - long-term memory (persistent storage)

### 3. Learning Mechanism
- Store interactions
- Track:
  - successful responses
  - failed responses
- Improve future outputs based on:
  - similarity matching
  - reinforcement signals

---

## Model Strategy

Use lightweight AI:

### Option A (Preferred)
- TensorFlow.js
- Use small pretrained models OR simple embeddings

### Option B
- Brain.js
- Use for pattern learning and classification

DO NOT:
- Attempt to replicate GPT/Claude scale models
- Use fake “training” loops with no real improvement

---

## Intelligence Layer (Important)

Simulate reasoning using:

1. Context retrieval
2. Pattern matching
3. Rule-based augmentation

Flow:

User Input  
→ Search memory  
→ Retrieve similar cases  
→ Generate response  
→ Store result  

---

## Self-Improvement Loop

After each interaction:

1. Save input + output
2. Evaluate response quality (basic heuristic)
3. Adjust:
   - weights (if using Brain.js)
   - stored examples
4. Improve matching for next time

---

## Tools Integration

Allow the AI to:

- Read/write files
- Access project code
- Analyze simple logic

But:

- NEVER modify core game logic automatically
- Suggest changes instead of applying blindly

---

## Backend Requirements

- Create a Node.js service:
  - `/api/ai/chat`
  - `/api/ai/memory`
- Store memory in:
  - SQLite OR JSON file

---

## Frontend (Optional but recommended)

- Simple chat UI
- Message history
- Debug panel:
  - show memory hits
  - show reasoning steps

---

## Safety Rules

- Do NOT expose sensitive backend data
- Do NOT override game logic
- Do NOT hallucinate system-level changes

---

## Goal Outcome

A working AI chatbot that:

- Responds intelligently
- Improves over time
- Learns from interactions
- Uses memory effectively
- Runs locally

---

## Important Note

This is NOT a true LLM.

This is a:
- hybrid AI system
- memory + logic + lightweight model

Focus on:
- usefulness
- stability
- incremental improvement

NOT:
- fake intelligence
- random responses
