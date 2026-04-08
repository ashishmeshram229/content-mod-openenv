"""
verify_local.py — Run this BEFORE pushing to HF to catch all issues.
Tests the exact same checks the OpenEnv validator runs.

Usage:
    # Terminal 1: start the server
    uvicorn app:app --host 0.0.0.0 --port 7860

    # Terminal 2: run this script
    python verify_local.py
"""

import json
import sys
import urllib.request
import urllib.error

BASE = "http://localhost:7860"
PASS = "✓"
FAIL = "✗"
results = []


def check(label, ok, detail=""):
    results.append((PASS if ok else FAIL, label, detail))
    print(f"  {'✓' if ok else '✗'}  {label}" + (f"  [{detail}]" if detail else ""))
    return ok


def post(path, body=None):
    """POST with optional body. Returns (status_code, response_dict)."""
    url = BASE + path
    if body is None:
        data = None
        headers = {}
    else:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
        except Exception:
            body = {"error": str(e)}
        return e.code, body
    except Exception as e:
        return 0, {"error": str(e)}


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


print("\n" + "="*60)
print("  OpenEnv Validator Simulation")
print("="*60)

# ── 1. Health check ──────────────────────────────────────────
print("\n── GET /health ─────────────────────────────────────────")
code, body = get("/health")
check("GET /health returns 200", code == 200, f"got {code}")
check("health has 'status'", "status" in body, str(body))

# ── 2. Root endpoint ─────────────────────────────────────────
print("\n── GET / ───────────────────────────────────────────────")
code, body = get("/")
check("GET / returns 200", code == 200, f"got {code}")

# ── 3. POST /reset — EXACTLY what validator sends ────────────
print("\n── POST /reset (validator scenarios) ───────────────────")

# Scenario A: No body at all (null body)
code, body = post("/reset", body=None)
check("POST /reset (no body) → 200", code == 200, f"got {code}: {str(body)[:80]}")
check("reset (no body) has 'content'", "content" in body, str(body)[:80])

# Scenario B: Empty JSON {}
code, body = post("/reset", body={})
check("POST /reset ({}) → 200", code == 200, f"got {code}: {str(body)[:80]}")
check("reset ({}) has 'content'", "content" in body, str(body)[:80])

# Scenario C: With task
code, body = post("/reset", body={"task": "easy"})
check("POST /reset ({'task':'easy'}) → 200", code == 200, f"got {code}")
if "content" in body:
    check("observation has content_id", "content_id" in body["content"],
          str(body["content"])[:60])
    check("observation has queue_total", "queue_total" in body, "")
    first_content_id = body["content"]["content_id"]
else:
    check("observation has content_id", False, "no 'content' key in response")
    first_content_id = "e001"

# Medium and hard
for task in ["medium", "hard"]:
    code, body = post("/reset", body={"task": task})
    check(f"POST /reset (task={task}) → 200", code == 200, f"got {code}")

# ── 4. POST /step ────────────────────────────────────────────
print("\n── POST /step ──────────────────────────────────────────")

# Reset first
post("/reset", body={"task": "easy"})

action = {
    "content_id": first_content_id,
    "decision": "remove",
    "violation_category": "spam",
    "severity_assessment": "medium",
    "confidence": 0.9,
    "reasoning": "test"
}

code, body = post("/step", body={"action": action})
check("POST /step → 200", code == 200, f"got {code}")
check("step response has 'observation'", "observation" in body, str(body)[:60])
check("step response has 'reward'", "reward" in body, "")
check("step response has 'done'", "done" in body, "")
check("step response has 'info'", "info" in body, "")
if "reward" in body:
    r = body["reward"]
    check("reward is in [-1, 1]", -1.0 <= r <= 1.0, f"reward={r}")

# ── 5. GET /state ────────────────────────────────────────────
print("\n── GET /state ──────────────────────────────────────────")
code, body = get("/state")
check("GET /state → 200", code == 200, f"got {code}")
check("state has 'task_id'", "task_id" in body, str(body)[:60])

# ── 6. Full episode (easy) ───────────────────────────────────
print("\n── Full easy episode ───────────────────────────────────")
code, obs = post("/reset", body={"task": "easy"})
episode_ok = code == 200
steps = 0
done = False
while not done and steps < 20:
    cid = obs.get("content", {}).get("content_id", "unknown")
    action = {
        "content_id": cid,
        "decision": "escalate",
        "violation_category": "none",
        "severity_assessment": "medium",
        "confidence": 0.5,
        "reasoning": "verify test"
    }
    code, result = post("/step", body={"action": action})
    if code != 200:
        episode_ok = False
        break
    done = result.get("done", False)
    obs = result.get("observation", obs)
    steps += 1

check("Full easy episode completes", episode_ok and done, f"{steps} steps, done={done}")
if done:
    ep = result.get("info", {}).get("episode_result", {})
    score = ep.get("final_score", -1)
    check("Episode result has final_score in [0,1]",
          0.0 <= score <= 1.0, f"score={score}")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
print(f"  Passed: {passed}  |  Failed: {failed}")

if failed == 0:
    print("\n  🎉 ALL CHECKS PASSED — safe to push to HF!")
else:
    print(f"\n  ❌ {failed} check(s) failed — fix before pushing.")
    print("\n  Failed checks:")
    for s, label, detail in results:
        if s == FAIL:
            print(f"    ✗ {label}  [{detail}]")
print("="*60)

sys.exit(0 if failed == 0 else 1)