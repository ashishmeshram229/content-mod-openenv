"""
inference.py - Content Moderation OpenEnv Baseline Agent
MANDATORY VARIABLES: API_BASE_URL, MODEL_NAME, HF_TOKEN
"""
import json
import os
import sys
import textwrap
import time
import urllib.request
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.3-70B-Instruct")
API_KEY      = (os.getenv("HF_TOKEN") or os.getenv("API_KEY") or
                os.getenv("OPENAI_API_KEY") or "hf_placeholder")
ENV_URL      = os.getenv("ENV_BASE_URL", "https://heist-content-mod-openenv.hf.space")
TEMPERATURE  = 0.0
MAX_TOKENS   = 512
TASKS        = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# Structured output helpers — required by Phase 2 validator
# ---------------------------------------------------------------------------
def log_start(task: str):
    print(f"[START] task={task}", flush=True)

def log_step(task: str, step: int, reward: float, done: bool):
    print(f"[STEP] task={task} step={step} reward={reward:.4f} done={done}", flush=True)

def log_end(task: str, score: float, steps: int):
    print(f"[END] task={task} score={score:.4f} steps={steps}", flush=True)

# ---------------------------------------------------------------------------
# OpenAI client — inside function so module-level crash is impossible
# ---------------------------------------------------------------------------
def make_client():
    try:
        from openai import OpenAI
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[WARNING] OpenAI client failed: {e}", flush=True)
        return None

# ---------------------------------------------------------------------------
# Env HTTP helpers
# ---------------------------------------------------------------------------
def _post(path, body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        ENV_URL.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

def _get(path):
    with urllib.request.urlopen(ENV_URL.rstrip("/") + path, timeout=30) as r:
        return json.loads(r.read())

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM = textwrap.dedent("""
You are an expert content moderator. Respond with ONLY valid JSON:
{
  "content_id": "<id>",
  "decision": "<approve|remove|escalate|warn_user|age_restrict|shadow_ban>",
  "violation_category": "<hate_speech|harassment|misinformation|spam|explicit_content|violence|self_harm|none>",
  "severity_assessment": "<low|medium|high|critical>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief>",
  "policy_rule_cited": null
}
Rules: approve=fine, remove=clear violation, escalate=ambiguous,
warn_user=borderline, age_restrict=adult but legal, shadow_ban=spam.
Self-harm ALWAYS needs action. No markdown, JSON only.
""").strip()

def build_prompt(obs):
    c = obs.get("content", {})
    u = obs.get("user_context", {})
    rules = "\n".join(f"  [{r['rule_id']}] {r['description']}"
                      for r in obs.get("applicable_rules", []))
    return (f"content_id: {c.get('content_id')}\n"
            f"text: {c.get('text','')}\n"
            f"reported: {c.get('reported_count')} views: {c.get('view_count')}\n"
            f"user_reputation: {u.get('reputation')} "
            f"prior_violations: {u.get('prior_violations',0)} "
            f"verified: {u.get('is_verified',False)}\n"
            f"queue: {obs.get('queue_position')}/{obs.get('queue_total')}\n"
            f"rules:\n{rules}\nJSON only.")

def call_llm(client, prompt):
    if client is None:
        return ""
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user",   "content": prompt}],
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"  [LLM] {e}", flush=True)
        return ""

def parse_action(text, content_id):
    try:
        t = text.strip()
        if t.startswith("```"):
            lines = t.split("\n")
            t = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        p = json.loads(t)
        p["content_id"] = content_id
        p.setdefault("reasoning", "")
        p.setdefault("policy_rule_cited", None)
        p.setdefault("escalation_note", None)
        p["confidence"] = float(max(0.0, min(1.0, p.get("confidence", 0.5))))
        return p
    except Exception:
        return {"content_id": content_id, "decision": "escalate",
                "violation_category": "none", "severity_assessment": "medium",
                "confidence": 0.1, "reasoning": "fallback",
                "policy_rule_cited": None, "escalation_note": None}

# ---------------------------------------------------------------------------
# Run one task — prints [START] [STEP] [END] blocks
# ---------------------------------------------------------------------------
def run_task(task_name, client):
    # [START] block — required by validator
    log_start(task_name)

    try:
        obs = _post("/reset", {"task": task_name})
    except Exception as e:
        print(f"  [ERROR] reset failed: {e}", flush=True)
        log_end(task_name, 0.0, 0)
        return {"task": task_name, "final_score": 0.0, "accuracy": 0.0,
                "correct": 0, "total": 0, "error": str(e)}

    total_reward = 0.0
    step = 0
    episode_result = {}

    while True:
        try:
            cid    = obs["content"]["content_id"]
            action = parse_action(call_llm(client, build_prompt(obs)), cid)
            result = _post("/step", {"action": action})
            obs    = result["observation"]
            reward = float(result["reward"])
            done   = bool(result["done"])
            total_reward += reward
            step += 1

            # [STEP] block — required by validator
            log_step(task_name, step, reward, done)

            if done:
                episode_result = result["info"].get("episode_result", {})
                break

        except Exception as e:
            print(f"  [ERROR] step {step}: {e}", flush=True)
            break

    score = float(episode_result.get("final_score", 0.0))

    # [END] block — required by validator
    log_end(task_name, score, step)

    return {
        "task":           task_name,
        "final_score":    score,
        "accuracy":       episode_result.get("accuracy", 0.0),
        "correct":        episode_result.get("correct_decisions", 0),
        "total":          episode_result.get("total_items", 0),
        "total_reward":   round(total_reward, 4),
        "episode_result": episode_result,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[INFO] Content Moderation OpenEnv — Baseline Agent", flush=True)
    print(f"[INFO] ENV={ENV_URL} API={API_BASE_URL} MODEL={MODEL_NAME}", flush=True)

    client = make_client()

    try:
        h = _get("/health")
        print(f"[INFO] Env health: {h.get('status','unknown')}", flush=True)
    except Exception as e:
        print(f"[WARNING] Env health check failed: {e}", flush=True)

    results = []
    t0 = time.time()

    for task in TASKS:
        try:
            results.append(run_task(task, client))
        except Exception as e:
            print(f"[ERROR] Task {task} crashed: {e}", flush=True)
            log_start(task)
            log_end(task, 0.0, 0)
            results.append({"task": task, "final_score": 0.0,
                            "accuracy": 0.0, "correct": 0,
                            "total": 0, "error": str(e)})

    elapsed = time.time() - t0
    avg = sum(r.get("final_score", 0) for r in results) / len(results)

    print(f"[SUMMARY] avg_score={avg:.4f} runtime={elapsed:.1f}s", flush=True)
    for r in results:
        print(f"[RESULT] task={r['task']} score={r.get('final_score',0):.4f} "
              f"accuracy={r.get('accuracy',0):.4f} "
              f"correct={r.get('correct',0)}/{r.get('total',0)}", flush=True)

    output = {"model": MODEL_NAME, "results": results,
              "avg_score": round(avg, 4), "runtime_sec": round(elapsed, 1)}
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("[INFO] Saved baseline_results.json", flush=True)


if __name__ == "__main__":
    main()