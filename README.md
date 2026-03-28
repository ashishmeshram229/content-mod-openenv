---
title: Content Moderation OpenEnv
emoji: 🛡️
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - content-moderation
  - trust-and-safety
  - rl-environment
  - agent-evaluation
---

# 🛡️ Content Moderation OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0.0-blue)](https://github.com/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

A **real-world content moderation environment** for training and evaluating AI agents on trust & safety decision-making. Built for the OpenEnv Hackathon (Meta × PyTorch × Hugging Face).

---

## Problem Description & Motivation

Content moderation is one of the most consequential decision-making tasks performed at scale on the internet. Human moderators at platforms like Meta, YouTube, and Twitter review millions of posts per day, making split-second decisions that affect:

- **Free expression** — wrongly removing legitimate speech
- **User safety** — failing to catch harassment, self-harm content, or misinformation
- **Platform integrity** — allowing spam bots and coordinated inauthentic behaviour

This environment simulates the exact workflow of a trust & safety queue: an agent observes a piece of user-generated content with full context (user reputation, report counts, applicable policies), then selects an enforcement action from a structured action space.

**Why this is ideal for agent evaluation:**
- Decisions have clear right/wrong answers grounded in policy
- Intermediate signals are available (severity, category, confidence)
- Difficulty scales naturally (clear violations → adversarial evasion)
- Consistency across similar cases is measurable

---

## Environment Overview

```
ContentModerationEnv
├── Observation: content item + user context + policy rules + queue state
├── Action:      enforcement decision + violation category + severity + reasoning
├── Reward:      shaped signal (decision correctness, category, severity, efficiency)
└── Episode:     a queue of N items to moderate (10–15 depending on task)
```

**Environment ID:** `content-moderation-v1`  
**Max steps per episode:** 50  
**Reward range:** [-1.0, 1.0]

---

## Action Space

| Field | Type | Options |
|---|---|---|
| `decision` | enum | `approve`, `remove`, `escalate`, `warn_user`, `age_restrict`, `shadow_ban` |
| `violation_category` | enum | `hate_speech`, `harassment`, `misinformation`, `spam`, `explicit_content`, `violence`, `self_harm`, `none` |
| `severity_assessment` | enum | `low`, `medium`, `high`, `critical` |
| `confidence` | float [0,1] | Agent's confidence in the decision |
| `reasoning` | string | Optional chain-of-thought |
| `policy_rule_cited` | string | Reference to applicable rule (P001–P010) |

---

## Observation Space

Each observation contains:

| Field | Description |
|---|---|
| `content` | The post: text, type, platform, reports, views, metadata |
| `user_context` | Author reputation, account age, prior violations, verified status |
| `applicable_rules` | List of platform policies that may apply |
| `queue_position` | Current item position in the episode queue |
| `backlog_pressure` | Float [0,1] — how full the queue is |
| `correct_so_far` | Running count of correct decisions this episode |
| `recent_decisions` | Last 3 decisions (for consistency awareness) |

---

## Task Descriptions

### 🟢 Easy — Clear-cut Moderation (10 items)
Unambiguous violations: spam bots, explicit hate speech, obvious self-harm content, and clearly benign posts. No contextual nuance required. Expected agent accuracy: **≥0.90**.

### 🟡 Medium — Contextual Moderation (12 items)
Cases where context determines the decision: satire vs. genuine hate, journalism vs. violence glorification, crisis signals, unverified health claims, and coordinated spam. Expected accuracy: **≥0.75**.

### 🔴 Hard — Adversarial Moderation (15 items)
Deliberately crafted to challenge frontier models:
- **Coded language** ("lose oxygen permanently 😉")
- **Character substitution** to evade keyword filters
- **Verified user misinformation** (doctor with 200k followers)
- **Policy conflicts** (journalism exception vs. violence policy)
- **Banned user appeals** (legitimate vs. bad-faith)
- **Disguised eating disorder content** framed as "wellness"

Expected accuracy for a frontier model: **≥0.65**.

---

## Reward Design

The reward function provides **shaped signals** throughout the episode, not just at the end:

```
reward = 0.60 × decision_score
       + 0.25 × category_score
       + 0.15 × severity_score
       + efficiency_bonus      (+0.10 for not over-escalating benign content)
       + consistency_bonus     (+0.05 for consistent policy application)
       + penalties
```

**Penalties:**
| Situation | Penalty |
|---|---|
| Approved self-harm/critical content | -0.50 to -0.80 |
| Removed clearly benign content | -0.20 |
| Wrong content_id submitted | -0.30 |
| Low confidence (< 0.5) | -0.05 per 0.1 below threshold |

**Partial credit:** Decisions that are "close" receive partial scores (e.g., `warn_user` when `remove` was expected = 0.4 × 0.60 = 0.24 decision score).

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start new episode: `{"task": "easy"}` |
| `POST` | `/step` | Submit action: `{"action": {...}}` |
| `GET` | `/state` | Full environment state |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive API docs (Swagger) |

---

## Setup Instructions

### Local Development

```bash
git clone https://github.com/YOUR_USERNAME/content-mod-openenv
cd content-mod-openenv

pip install -r requirements.txt

# Start the server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload

# Run inference (in another terminal)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

### Docker

```bash
docker build -t content-mod-openenv .
docker run -p 7860:7860 content-mod-openenv

# With inference
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct" \
  -e HF_TOKEN="hf_xxx" \
  content-mod-openenv
```

### Quick API Test

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'

# Submit a decision
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "content_id": "e001",
      "decision": "remove",
      "violation_category": "spam",
      "severity_assessment": "medium",
      "confidence": 0.95,
      "reasoning": "Clear spam post with external link from new account"
    }
  }'
```

---

## Usage Guide

### Python SDK

```python
from env.environment import ContentModerationEnv
from env.models import ModerationAction, ModerationDecision, ViolationCategory, Severity

env = ContentModerationEnv()

# Run easy task
obs = env.reset(task="easy")
print(f"First item: {obs.content.text[:80]}")

action = ModerationAction(
    content_id=obs.content.content_id,
    decision=ModerationDecision.REMOVE,
    violation_category=ViolationCategory.SPAM,
    severity_assessment=Severity.MEDIUM,
    confidence=0.95,
    reasoning="Clear spam from new account",
)

obs, reward, done, info = env.step(action)
print(f"Reward: {reward:.4f}")
print(f"Breakdown: {info['reward_breakdown']['explanation']}")
```

---

## Baseline Results

Measured with `meta-llama/Llama-3.3-70B-Instruct` via Hugging Face router:

| Task | Score | Accuracy | Correct/Total |
|---|---|---|---|
| Easy | ~0.82 | ~0.80 | 8/10 |
| Medium | ~0.68 | ~0.67 | 8/12 |
| Hard | ~0.54 | ~0.53 | 8/15 |
| **Average** | **~0.68** | | |

*Baseline scores are reproducible with `TEMPERATURE=0.0` and fixed model.*

---

## Project Structure

```
content-mod-openenv/
├── env/
│   ├── __init__.py
│   ├── environment.py    # step() / reset() / state()
│   ├── models.py         # Pydantic schemas
│   ├── tasks.py          # 37 content items across 3 tasks
│   └── grader.py         # Deterministic grader functions
├── app.py                # FastAPI server
├── inference.py          # Baseline agent (OpenAI client)
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
