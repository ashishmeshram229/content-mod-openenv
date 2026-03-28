"""
inference.py — Content Moderation OpenEnv Baseline Agent
=========================================================
MANDATORY VARIABLES (set in environment before running):
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key

Usage:
    API_BASE_URL=https://router.huggingface.co/v1 \\
    MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \\
    HF_TOKEN=hf_xxx \\
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str       = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str            = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME: str         = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_BASE_URL: str       = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TEMPERATURE: float      = 0.0   # deterministic
MAX_TOKENS: int         = 512
TASKS: List[str]        = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# HTTP client for the environment
# ---------------------------------------------------------------------------

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def _env_post(path: str, payload: Dict) -> Dict:
    import urllib.request, json as _json
    data = _json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{ENV_BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return _json.loads(resp.read())


def _env_get(path: str) -> Dict:
    import urllib.request, json as _json
    with urllib.request.urlopen(f"{ENV_BASE_URL}{path}", timeout=10) as resp:
        return _json.loads(resp.read())


def env_reset(task: str) -> Dict:
    return _env_post("/reset", {"task": task})


def env_step(action: Dict) -> Dict:
    return _env_post("/step", {"action": action})


def env_state() -> Dict:
    return _env_get("/state")


# ---------------------------------------------------------------------------
# In-process fallback (no server needed)
# ---------------------------------------------------------------------------

def _get_local_env():
    """Use environment directly without HTTP server (for self-contained runs)."""
    try:
        from env.environment import ContentModerationEnv
        from env.models import ModerationAction
        return ContentModerationEnv(), ModerationAction
    except ImportError:
        return None, None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert content moderator for a large social media platform.
Your task is to review user-generated content and make policy enforcement decisions.

For each content item, you MUST respond with a valid JSON object (no markdown, no explanation outside the JSON):

{
  "content_id": "<exact content_id from the observation>",
  "decision": "<one of: approve | remove | escalate | warn_user | age_restrict | shadow_ban>",
  "violation_category": "<one of: hate_speech | harassment | misinformation | spam | explicit_content | violence | self_harm | none>",
  "severity_assessment": "<one of: low | medium | high | critical>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation of your decision>",
  "policy_rule_cited": "<rule_id if applicable, else null>"
}

Decision guide:
- approve: content is fine, no action needed
- remove: content clearly violates policy and must be taken down
- escalate: ambiguous or high-stakes case needing human review
- warn_user: content is borderline; keep but flag the user's account
- age_restrict: content is adult/sensitive but legal; limit audience
- shadow_ban: suppress reach without notifying user (spam/inauthentic)

Key considerations:
1. User reputation and prior violations matter
2. Verified journalists may report on violence without removal
3. Satire with clear labelling is permitted
4. Self-harm content always requires action (remove or escalate)
5. Context distinguishes education from promotion
6. Coded language and character substitution (l33tspeak) still violates policy
""").strip()


def build_user_prompt(obs: Dict) -> str:
    content    = obs.get("content", {})
    user_ctx   = obs.get("user_context", {})
    rules      = obs.get("applicable_rules", [])
    recent     = obs.get("recent_decisions", [])

    rule_lines = "\n".join(
        f"  [{r['rule_id']}] {r['description']} (severity: {r['severity']})"
        for r in rules
    )
    recent_lines = (
        "\n".join(
            f"  - {d['content_id']}: {d['decision']} | {d['category']}"
            for d in recent
        )
        or "  None yet"
    )

    return textwrap.dedent(f"""
    === Content Item to Moderate ===
    content_id: {content.get('content_id')}
    type: {content.get('content_type')}
    text: {content.get('text', '[no text]')}
    platform: {content.get('platform')}
    language: {content.get('language')}
    reported_count: {content.get('reported_count')}
    view_count: {content.get('view_count')}
    engagement_rate: {content.get('engagement_rate')}
    has_links: {content.get('has_links')}
    metadata: {json.dumps(content.get('metadata', {}))}

    === User Context ===
    user_id: {user_ctx.get('user_id')}
    reputation: {user_ctx.get('reputation')}
    account_age_days: {user_ctx.get('account_age_days')}
    prior_violations: {user_ctx.get('prior_violations', 0)}
    follower_count: {user_ctx.get('follower_count', 0)}
    is_verified: {user_ctx.get('is_verified', False)}

    === Queue Status ===
    item {obs.get('queue_position')} of {obs.get('queue_total')}
    backlog_pressure: {obs.get('backlog_pressure')}
    correct_so_far: {obs.get('correct_so_far')} / {obs.get('items_processed')}

    === Applicable Policy Rules ===
{rule_lines}

    === Recent Decisions (for consistency) ===
{recent_lines}

    Respond with a single valid JSON object only.
    """).strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"  [LLM ERROR] {exc}")
        return ""


def parse_action(response_text: str, content_id: str) -> Dict[str, Any]:
    """Parse LLM response into a ModerationAction dict."""
    # Strip markdown code blocks if present
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        parsed = json.loads(text)
        # Ensure content_id matches
        parsed["content_id"] = content_id
        # Defaults for optional fields
        parsed.setdefault("reasoning", "")
        parsed.setdefault("escalation_note", None)
        parsed.setdefault("policy_rule_cited", None)
        # Clamp confidence
        parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        return parsed
    except (json.JSONDecodeError, ValueError):
        print(f"  [PARSE ERROR] Could not parse: {response_text[:200]}")
        # Fallback: escalate with low confidence
        return {
            "content_id":          content_id,
            "decision":            "escalate",
            "violation_category":  "none",
            "severity_assessment": "medium",
            "confidence":          0.1,
            "reasoning":           "Parse error — defaulting to escalate.",
            "escalation_note":     "LLM parse failure",
            "policy_rule_cited":   None,
        }


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task_local(
    task_name: str,
    client: OpenAI,
    env,
    ModerationAction,
) -> Dict[str, Any]:
    """Run a task using the environment directly (no HTTP server)."""
    print(f"\n{'='*60}")
    print(f"  TASK: {task_name.upper()}")
    print(f"{'='*60}")

    obs_model = env.reset(task=task_name)
    obs = obs_model.dict()

    total_reward = 0.0
    step = 0
    episode_result = None

    while True:
        content_id = obs["content"]["content_id"]
        print(f"\n  Step {step+1} | Item: {content_id} | "
              f"Queue: {obs['queue_position']}/{obs['queue_total']}")

        prompt   = build_user_prompt(obs)
        response = call_llm(client, prompt)
        action_dict = parse_action(response, content_id)

        print(f"  Decision: {action_dict['decision']} | "
              f"Category: {action_dict['violation_category']} | "
              f"Severity: {action_dict['severity_assessment']} | "
              f"Conf: {action_dict['confidence']:.2f}")

        action = ModerationAction(**action_dict)
        obs_model, reward, done, info = env.step(action)
        obs = obs_model.dict()
        total_reward += reward
        step += 1

        print(f"  Reward: {reward:+.4f} | Cumulative: {total_reward:+.4f}")

        if done:
            episode_result = info.get("episode_result", {})
            break

    print(f"\n  --- {task_name.upper()} COMPLETE ---")
    print(f"  Final score:       {episode_result.get('final_score', 0):.4f}")
    print(f"  Accuracy:          {episode_result.get('accuracy', 0):.4f}")
    print(f"  Correct decisions: {episode_result.get('correct_decisions')}/{episode_result.get('total_items')}")

    return {
        "task":           task_name,
        "final_score":    episode_result.get("final_score", 0),
        "accuracy":       episode_result.get("accuracy", 0),
        "correct":        episode_result.get("correct_decisions", 0),
        "total":          episode_result.get("total_items", 0),
        "total_reward":   round(total_reward, 4),
        "episode_result": episode_result,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.")
        sys.exit(1)

    print(f"\nContent Moderation OpenEnv — Baseline Agent")
    print(f"  API Base URL : {API_BASE_URL}")
    print(f"  Model        : {MODEL_NAME}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Try in-process first (no server dependency)
    env, ModerationActionClass = _get_local_env()

    all_results: List[Dict] = []
    start_time = time.time()

    for task_name in TASKS:
        if env is not None:
            result = run_task_local(task_name, client, env, ModerationActionClass)
        else:
            print(f"[WARNING] Local env not available. Skipping {task_name}.")
            result = {"task": task_name, "final_score": 0.0, "error": "env_unavailable"}
        all_results.append(result)

    elapsed = time.time() - start_time

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Score':>8} {'Accuracy':>10} {'Correct':>10}")
    print(f"  {'-'*40}")
    for r in all_results:
        score    = r.get("final_score", 0)
        accuracy = r.get("accuracy", 0)
        correct  = r.get("correct", "?")
        total    = r.get("total", "?")
        print(f"  {r['task']:<10} {score:>8.4f} {accuracy:>10.4f} {str(correct)+'/'+str(total):>10}")

    avg_score = sum(r.get("final_score", 0) for r in all_results) / len(all_results)
    print(f"\n  Average score: {avg_score:.4f}")
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"{'='*60}")

    # Write JSON results
    output = {
        "model":       MODEL_NAME,
        "results":     all_results,
        "avg_score":   round(avg_score, 4),
        "runtime_sec": round(elapsed, 1),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\n  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
