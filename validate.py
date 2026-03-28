"""
validate.py — Pre-submission OpenEnv compliance validator.
Run this before submitting to catch any issues.

Usage:
    python validate.py
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict, List, Tuple

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

results: List[Tuple[str, str, str]] = []   # (status, check, detail)


def check(label: str, condition: bool, detail: str = "", warn_only: bool = False) -> bool:
    status = PASS if condition else (WARN if warn_only else FAIL)
    results.append((status, label, detail))
    return condition


# ---------------------------------------------------------------------------
# 1. File presence
# ---------------------------------------------------------------------------

import os

REQUIRED_FILES = [
    "inference.py",
    "openenv.yaml",
    "Dockerfile",
    "README.md",
    "requirements.txt",
    "env/environment.py",
    "env/models.py",
    "env/tasks.py",
    "env/grader.py",
    "env/__init__.py",
    "app.py",
    "LICENSE",
]

print("\n── File presence ──────────────────────────────────────")
for f in REQUIRED_FILES:
    check(f"File exists: {f}", os.path.isfile(f))

# ---------------------------------------------------------------------------
# 2. openenv.yaml
# ---------------------------------------------------------------------------

print("\n── openenv.yaml ───────────────────────────────────────")
try:
    import yaml
    with open("openenv.yaml") as fh:
        spec = yaml.safe_load(fh)

    check("Has 'name' field",        "name"    in spec, spec.get("name", ""))
    check("Has 'version' field",     "version" in spec, spec.get("version", ""))
    check("Has 'tasks' list",        "tasks"   in spec and len(spec["tasks"]) >= 3,
          f"{len(spec.get('tasks', []))} task(s)")
    check("Has 'openenv' tag",       "openenv" in spec.get("tags", []))
    check("Has 'reward' section",    "reward"  in spec)
    check("Has 'endpoints' section", "endpoints" in spec)

    tasks = spec.get("tasks", [])
    difficulties = [t.get("difficulty") for t in tasks]
    check("Has easy task",   "easy"   in difficulties)
    check("Has medium task", "medium" in difficulties)
    check("Has hard task",   "hard"   in difficulties)
except Exception as e:
    check("openenv.yaml parseable", False, str(e))

# ---------------------------------------------------------------------------
# 3. Pydantic models
# ---------------------------------------------------------------------------

print("\n── Pydantic models ────────────────────────────────────")
try:
    from env.models import (
        ModerationAction, ModerationObservation, ModerationReward,
        ModerationDecision, ViolationCategory, Severity,
        ContentItem, UserContext, PolicyRule,
    )
    check("ModerationObservation importable", True)
    check("ModerationAction importable",      True)
    check("ModerationReward importable",      True)

    # Instantiation test
    from env.models import ContentType, UserReputation
    content = ContentItem(
        content_id="v001", content_type=ContentType.TEXT,
        text="test", reported_count=0, view_count=100, engagement_rate=0.1, has_links=False,
    )
    user = UserContext(user_id="u001", reputation=UserReputation.TRUSTED, account_age_days=100)
    obs = ModerationObservation(
        content=content, user_context=user, applicable_rules=[],
        queue_position=1, queue_total=5, items_processed=0,
        correct_so_far=0, step_number=0, task_id="test", task_difficulty="easy",
    )
    check("ModerationObservation instantiates", True)

    action = ModerationAction(
        content_id="v001",
        decision=ModerationDecision.APPROVE,
        violation_category=ViolationCategory.NONE,
        severity_assessment=Severity.LOW,
        confidence=0.9,
    )
    check("ModerationAction instantiates", True)

    reward = ModerationReward(
        decision_score=0.6, category_score=0.25, severity_score=0.15,
        efficiency_bonus=0.0, consistency_bonus=0.0, penalty=0.0,
        total=1.0, explanation="test",
    )
    check("ModerationReward instantiates", True)

except Exception as e:
    check("Models import/instantiate", False, traceback.format_exc(limit=3))

# ---------------------------------------------------------------------------
# 4. Environment API — reset / step / state
# ---------------------------------------------------------------------------

print("\n── Environment API ────────────────────────────────────")
try:
    from env.environment import ContentModerationEnv
    env = ContentModerationEnv()
    check("ContentModerationEnv instantiates", True)

    for task_name in ["easy", "medium", "hard"]:
        obs = env.reset(task=task_name)
        check(f"reset('{task_name}') returns ModerationObservation",
              isinstance(obs, ModerationObservation),
              f"type={type(obs).__name__}")

        # Step with a valid action
        action = ModerationAction(
            content_id=obs.content.content_id,
            decision=ModerationDecision.ESCALATE,
            violation_category=ViolationCategory.NONE,
            severity_assessment=Severity.LOW,
            confidence=0.5,
        )
        result = env.step(action)
        check(f"step() returns 4-tuple for '{task_name}'", len(result) == 4)
        _obs, reward, done, info = result
        check(f"reward is float for '{task_name}'",     isinstance(reward, float))
        check(f"done is bool for '{task_name}'",        isinstance(done, bool))
        check(f"info is dict for '{task_name}'",        isinstance(info, dict))
        check(f"reward in [-1, 1] for '{task_name}'",  -1.0 <= reward <= 1.0,
              f"reward={reward:.4f}")

        state = env.state()
        check(f"state() returns dict for '{task_name}'", isinstance(state, dict))
        check(f"state() has 'task_id' for '{task_name}'", "task_id" in state)

except Exception as e:
    check("Environment API", False, traceback.format_exc(limit=5))

# ---------------------------------------------------------------------------
# 5. Graders
# ---------------------------------------------------------------------------

print("\n── Graders ────────────────────────────────────────────")
try:
    from env.grader import GRADERS, grade_single_action
    from env.tasks import ALL_TASKS

    for difficulty, grader_fn in GRADERS.items():
        task_def = ALL_TASKS[difficulty]
        items    = task_def["items"]

        # Build a set of decisions using ground-truth answers
        decisions = [
            {
                "action": ModerationAction(
                    content_id=item["content"].content_id,
                    decision=item["gt_decision"],
                    violation_category=item["gt_category"],
                    severity_assessment=item["gt_severity"],
                    confidence=0.95,
                ),
                "gt_decision": item["gt_decision"],
                "gt_category": item["gt_category"],
                "gt_severity": item["gt_severity"],
            }
            for item in items
        ]
        result = grader_fn(decisions)
        check(f"grade_{difficulty}() returns EpisodeResult", hasattr(result, "final_score"))
        check(f"grade_{difficulty}() score in [0,1]",
              0.0 <= result.final_score <= 1.0, f"score={result.final_score:.4f}")
        check(f"grade_{difficulty}() score is deterministic",
              grader_fn(decisions).final_score == result.final_score)

        # Perfect run should score > 0.9
        check(f"grade_{difficulty}() perfect run scores >0.90",
              result.final_score >= 0.90, f"got {result.final_score:.4f}", warn_only=True)

        # Random (bad) decisions should score lower than perfect
        bad_decisions = [
            {
                "action": ModerationAction(
                    content_id=item["content"].content_id,
                    decision=ModerationDecision.APPROVE,
                    violation_category=ViolationCategory.NONE,
                    severity_assessment=Severity.LOW,
                    confidence=0.1,
                ),
                "gt_decision": item["gt_decision"],
                "gt_category": item["gt_category"],
                "gt_severity": item["gt_severity"],
            }
            for item in items
        ]
        bad_result = grader_fn(bad_decisions)
        check(f"grade_{difficulty}() bad run scores lower than perfect",
              bad_result.final_score < result.final_score,
              f"bad={bad_result.final_score:.4f} vs perfect={result.final_score:.4f}")

except Exception as e:
    check("Graders", False, traceback.format_exc(limit=5))

# ---------------------------------------------------------------------------
# 6. Task dataset integrity
# ---------------------------------------------------------------------------

print("\n── Task dataset ───────────────────────────────────────")
try:
    from env.tasks import ALL_TASKS

    for difficulty, task_def in ALL_TASKS.items():
        items = task_def["items"]
        check(f"Task '{difficulty}' has ≥3 items", len(items) >= 3, f"{len(items)} items")

        ids = [item["content"].content_id for item in items]
        check(f"Task '{difficulty}' all content_ids unique",
              len(ids) == len(set(ids)), f"ids={ids}")

        for item in items:
            assert "gt_decision" in item, f"Missing gt_decision in {item}"
            assert "gt_category" in item, f"Missing gt_category in {item}"
            assert "gt_severity" in item, f"Missing gt_severity in {item}"
        check(f"Task '{difficulty}' all items have ground truth", True)

except Exception as e:
    check("Task dataset integrity", False, traceback.format_exc(limit=3))

# ---------------------------------------------------------------------------
# 7. Dockerfile basics
# ---------------------------------------------------------------------------

print("\n── Dockerfile ─────────────────────────────────────────")
with open("Dockerfile") as fh:
    df = fh.read()
check("Dockerfile EXPOSE 7860",  "EXPOSE 7860"  in df)
check("Dockerfile HEALTHCHECK",  "HEALTHCHECK"  in df)
check("Dockerfile non-root user", "useradd"     in df or "USER" in df)
check("Dockerfile copies requirements", "COPY" in df and "requirements" in df)

# ---------------------------------------------------------------------------
# 8. inference.py compliance
# ---------------------------------------------------------------------------

print("\n── inference.py ───────────────────────────────────────")
with open("inference.py") as fh:
    inf = fh.read()
check("Uses 'from openai import OpenAI'", "from openai import OpenAI" in inf)
check("Reads API_BASE_URL env var",       "API_BASE_URL" in inf and "os.getenv" in inf)
check("Reads MODEL_NAME env var",         "MODEL_NAME"   in inf)
check("Reads HF_TOKEN env var",           "HF_TOKEN"     in inf)
check("Has TEMPERATURE constant",         "TEMPERATURE"  in inf)
check("Covers all 3 tasks",
      all(t in inf for t in ['"easy"', '"medium"', '"hard"']))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "="*56)
print("  VALIDATION SUMMARY")
print("="*56)
passed = sum(1 for s, _, _ in results if s == PASS)
warned = sum(1 for s, _, _ in results if s == WARN)
failed = sum(1 for s, _, _ in results if s == FAIL)

for status, label, detail in results:
    suffix = f"  [{detail}]" if detail else ""
    print(f"  {status}  {label}{suffix}")

print(f"\n  Passed: {passed}  |  Warnings: {warned}  |  Failed: {failed}")

if failed == 0:
    print("\n  🎉 ALL CHECKS PASSED — submission is ready!")
    sys.exit(0)
else:
    print(f"\n  ❌ {failed} check(s) failed — fix before submitting.")
    sys.exit(1)
