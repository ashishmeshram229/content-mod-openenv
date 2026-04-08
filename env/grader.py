"""
Deterministic grader functions for the Content Moderation environment.
Each grader produces a score in [0.0, 1.0] with no randomness.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from env.models import (
    EpisodeResult,
    ModerationAction,
    ModerationDecision,
    Severity,
    ViolationCategory,
)


# ---------------------------------------------------------------------------
# Decision compatibility matrix
# Partial credit for "close" decisions (e.g. REMOVE when WARN_USER expected)
# ---------------------------------------------------------------------------

# fmt: off
DECISION_SCORE: Dict[Tuple[ModerationDecision, ModerationDecision], float] = {
    # (predicted, ground_truth) -> score
    (ModerationDecision.APPROVE,      ModerationDecision.APPROVE):     1.0,
    (ModerationDecision.REMOVE,       ModerationDecision.REMOVE):      1.0,
    (ModerationDecision.ESCALATE,     ModerationDecision.ESCALATE):    1.0,
    (ModerationDecision.WARN_USER,    ModerationDecision.WARN_USER):   1.0,
    (ModerationDecision.AGE_RESTRICT, ModerationDecision.AGE_RESTRICT):1.0,
    (ModerationDecision.SHADOW_BAN,   ModerationDecision.SHADOW_BAN):  1.0,

    # Partial credit — conservative action when stricter was correct
    (ModerationDecision.WARN_USER,    ModerationDecision.REMOVE):      0.4,
    (ModerationDecision.ESCALATE,     ModerationDecision.REMOVE):      0.5,
    (ModerationDecision.AGE_RESTRICT, ModerationDecision.REMOVE):      0.3,

    # Partial credit — too strict when lenient was correct
    (ModerationDecision.REMOVE,       ModerationDecision.WARN_USER):   0.3,
    (ModerationDecision.REMOVE,       ModerationDecision.AGE_RESTRICT):0.3,
    (ModerationDecision.ESCALATE,     ModerationDecision.WARN_USER):   0.5,
    (ModerationDecision.ESCALATE,     ModerationDecision.AGE_RESTRICT):0.5,

    # Worst case — approving something that should be removed (or vice-versa)
    (ModerationDecision.APPROVE,      ModerationDecision.REMOVE):      0.0,
    (ModerationDecision.REMOVE,       ModerationDecision.APPROVE):     0.0,
    (ModerationDecision.APPROVE,      ModerationDecision.ESCALATE):    0.0,
    (ModerationDecision.APPROVE,      ModerationDecision.WARN_USER):   0.1,
    (ModerationDecision.APPROVE,      ModerationDecision.AGE_RESTRICT):0.1,
    (ModerationDecision.APPROVE,      ModerationDecision.SHADOW_BAN):  0.1,
}

SEVERITY_SCORE: Dict[Tuple[Severity, Severity], float] = {
    (Severity.LOW,      Severity.LOW):      1.0,
    (Severity.MEDIUM,   Severity.MEDIUM):   1.0,
    (Severity.HIGH,     Severity.HIGH):     1.0,
    (Severity.CRITICAL, Severity.CRITICAL): 1.0,
    # One level off
    (Severity.LOW,      Severity.MEDIUM):   0.5,
    (Severity.MEDIUM,   Severity.LOW):      0.5,
    (Severity.MEDIUM,   Severity.HIGH):     0.5,
    (Severity.HIGH,     Severity.MEDIUM):   0.5,
    (Severity.HIGH,     Severity.CRITICAL): 0.5,
    (Severity.CRITICAL, Severity.HIGH):     0.5,
    # Two levels off
    (Severity.LOW,      Severity.HIGH):     0.0,
    (Severity.HIGH,     Severity.LOW):      0.0,
    (Severity.LOW,      Severity.CRITICAL): 0.0,
    (Severity.CRITICAL, Severity.LOW):      0.0,
    (Severity.MEDIUM,   Severity.CRITICAL): 0.25,
    (Severity.CRITICAL, Severity.MEDIUM):   0.25,
}
# fmt: on


def _decision_score(predicted: ModerationDecision, gt: ModerationDecision) -> float:
    return DECISION_SCORE.get((predicted, gt), 0.2)


def _category_score(predicted: ViolationCategory, gt: ViolationCategory) -> float:
    if predicted == gt:
        return 1.0
    # Partial credit if none was predicted for a genuine violation (missed)
    if predicted == ViolationCategory.NONE and gt != ViolationCategory.NONE:
        return 0.0
    # Partial credit if violation detected but wrong category
    if predicted != ViolationCategory.NONE and gt != ViolationCategory.NONE:
        return 0.3
    # False positive (predicted violation on clean content)
    if predicted != ViolationCategory.NONE and gt == ViolationCategory.NONE:
        return 0.2
    return 0.0


def _severity_score(predicted: Severity, gt: Severity) -> float:
    return SEVERITY_SCORE.get((predicted, gt), 0.0)


# ---------------------------------------------------------------------------
# Per-item grader
# ---------------------------------------------------------------------------

def grade_single_action(
    action: ModerationAction,
    gt_decision: ModerationDecision,
    gt_category: ViolationCategory,
    gt_severity: Severity,
) -> Dict[str, float]:
    """
    Score a single moderation decision. Returns a dict of component scores.
    All scores in [0.0, 1.0].
    """
    d_score = _decision_score(action.decision, gt_decision)
    c_score = _category_score(action.violation_category, gt_category)
    s_score = _severity_score(action.severity_assessment, gt_severity)

    # Weighted composite for this item
    item_score = 0.60 * d_score + 0.25 * c_score + 0.15 * s_score

    return {
        "decision_score": d_score,
        "category_score": c_score,
        "severity_score": s_score,
        "item_score": round(item_score, 4),
    }


# ---------------------------------------------------------------------------
# Episode-level graders (one per task difficulty)
# ---------------------------------------------------------------------------

def _build_episode_result(
    task_id: str,
    difficulty: str,
    decisions: List[Dict[str, Any]],
) -> EpisodeResult:
    """
    decisions: list of dicts with keys:
        action (ModerationAction), gt_decision, gt_category, gt_severity
    """
    total = len(decisions)
    correct = 0
    item_scores: List[float] = []
    category_errors: Dict[str, int] = {}
    error_analysis: List[str] = []

    for d in decisions:
        action: ModerationAction = d["action"]
        gt_dec: ModerationDecision = d["gt_decision"]
        gt_cat: ViolationCategory  = d["gt_category"]
        gt_sev: Severity           = d["gt_severity"]

        scores = grade_single_action(action, gt_dec, gt_cat, gt_sev)
        item_scores.append(scores["item_score"])

        if scores["decision_score"] == 1.0:
            correct += 1
        else:
            err = (
                f"{action.content_id}: predicted={action.decision.value} "
                f"(expected={gt_dec.value}), "
                f"category={action.violation_category.value} "
                f"(expected={gt_cat.value})"
            )
            error_analysis.append(err)
            category_errors[gt_cat.value] = category_errors.get(gt_cat.value, 0) + 1

    accuracy = correct / total if total > 0 else 0.0
    avg_item_score = sum(item_scores) / len(item_scores) if item_scores else 0.0

    # Final score weights accuracy heavily but rewards partial credit
    final_score = round(0.60 * accuracy + 0.40 * avg_item_score, 4)
    final_score = max(0.0, min(1.0, final_score))

    return EpisodeResult(
        task_id=task_id,
        task_difficulty=difficulty,
        total_items=total,
        correct_decisions=correct,
        accuracy=round(accuracy, 4),
        avg_reward_per_step=round(avg_item_score, 4),
        final_score=final_score,
        category_breakdown={k: v for k, v in sorted(category_errors.items())},
        error_analysis=error_analysis[:10],  # cap for readability
    )


def grade_easy(decisions: List[Dict[str, Any]]) -> EpisodeResult:
    """Grader for the EASY task."""
    return _build_episode_result("task_easy_001", "easy", decisions)


def grade_medium(decisions: List[Dict[str, Any]]) -> EpisodeResult:
    """Grader for the MEDIUM task."""
    return _build_episode_result("task_medium_001", "medium", decisions)


def grade_hard(decisions: List[Dict[str, Any]]) -> EpisodeResult:
    """Grader for the HARD task."""
    return _build_episode_result("task_hard_001", "hard", decisions)


GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}
