"""
ContentModerationEnv — OpenEnv-compatible environment.

Simulates a real-world content moderation queue where an AI agent must
review user-generated content and decide on appropriate policy actions.

API:
    reset(task_id)    → ModerationObservation
    step(action)      → (ModerationObservation, float, bool, dict)
    state()           → dict
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple

from env.grader import GRADERS, grade_single_action
from env.models import (
    EpisodeResult,
    ModerationAction,
    ModerationDecision,
    ModerationObservation,
    ModerationReward,
    Severity,
    ViolationCategory,
)
from env.tasks import ALL_TASKS


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

R_CORRECT_DECISION   =  0.60   # full credit for correct approve/remove/etc.
R_CORRECT_CATEGORY   =  0.25   # full credit for correct violation category
R_CORRECT_SEVERITY   =  0.15   # full credit for correct severity
R_EFFICIENCY_BONUS   =  0.10   # bonus for not over-escalating benign content
R_CONSISTENCY_BONUS  =  0.05   # bonus for consistent policy application
R_MISS_CRITICAL      = -0.50   # penalty: approved content with CRITICAL severity
R_MISS_SELF_HARM     = -0.80   # extra penalty: missed self-harm
R_OVER_REMOVE        = -0.20   # penalty: removed clearly benign content
R_CONFIDENCE_PENALTY = -0.05   # penalty per 0.1 below 0.5 confidence (low quality)


class ContentModerationEnv:
    """
    OpenEnv environment for content moderation.
    """

    ENV_ID      = "content-moderation-v1"
    VERSION     = "1.0.0"
    MAX_STEPS   = 50   # hard cap per episode

    def __init__(self) -> None:
        self._task_id:       Optional[str]  = None
        self._difficulty:    Optional[str]  = None
        self._items:         List[Dict]     = []
        self._current_idx:   int            = 0
        self._step_number:   int            = 0
        self._done:          bool           = True
        self._decisions:     List[Dict]     = []
        self._cumulative_reward: float      = 0.0
        self._recent_decisions: List[Dict[str, str]] = []

        # Track consistency: same violation type should receive same decision
        self._violation_decision_history: Dict[str, str] = {}

    # -----------------------------------------------------------------------
    # reset
    # -----------------------------------------------------------------------

    def reset(self, task: str = "easy") -> ModerationObservation:
        """
        Reset the environment and return the first observation.
        task: "easy" | "medium" | "hard"
        """
        if task not in ALL_TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(ALL_TASKS)}")

        task_def = ALL_TASKS[task]
        self._task_id       = task_def["task_id"]
        self._difficulty    = task_def["difficulty"]
        self._items         = copy.deepcopy(task_def["items"])
        self._current_idx   = 0
        self._step_number   = 0
        self._done          = False
        self._decisions     = []
        self._cumulative_reward = 0.0
        self._recent_decisions  = []
        self._violation_decision_history = {}

        return self._make_observation()

    # -----------------------------------------------------------------------
    # step
    # -----------------------------------------------------------------------

    def step(
        self, action: ModerationAction
    ) -> Tuple[ModerationObservation, float, bool, Dict[str, Any]]:
        """
        Process one moderation decision.
        Returns: (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._current_idx >= len(self._items):
            raise RuntimeError("No more items in queue.")

        current_item_def = self._items[self._current_idx]
        self._step_number += 1

        # Validate that the action references the correct content item
        expected_id = current_item_def["content"].content_id
        if action.content_id != expected_id:
            # Penalise wrong content ID (agent is confused)
            reward_obj = ModerationReward(
                decision_score=0.0, category_score=0.0, severity_score=0.0,
                efficiency_bonus=0.0, consistency_bonus=0.0,
                penalty=-0.30,
                total=-0.30,
                explanation=f"Wrong content_id. Expected {expected_id}, got {action.content_id}.",
            )
            obs = self._make_observation()
            info = {"reward_breakdown": reward_obj.dict(), "error": "wrong_content_id"}
            return obs, reward_obj.total, self._done, info

        gt_decision = current_item_def["gt_decision"]
        gt_category = current_item_def["gt_category"]
        gt_severity = current_item_def["gt_severity"]

        # --- Compute component scores ---
        scores = grade_single_action(action, gt_decision, gt_category, gt_severity)

        # --- Efficiency bonus: don't over-escalate clearly benign content ---
        efficiency_bonus = 0.0
        if (gt_decision == ModerationDecision.APPROVE
                and action.decision == ModerationDecision.APPROVE
                and gt_category == ViolationCategory.NONE):
            efficiency_bonus = R_EFFICIENCY_BONUS

        # --- Consistency bonus ---
        consistency_bonus = 0.0
        vcat = action.violation_category.value
        if vcat != ViolationCategory.NONE.value:
            prev = self._violation_decision_history.get(vcat)
            if prev is None:
                self._violation_decision_history[vcat] = action.decision.value
            elif prev == action.decision.value:
                consistency_bonus = R_CONSISTENCY_BONUS

        # --- Penalties ---
        penalty = 0.0

        # Missed critical self-harm
        if (gt_category == ViolationCategory.SELF_HARM
                and gt_severity in (Severity.HIGH, Severity.CRITICAL)
                and action.decision == ModerationDecision.APPROVE):
            penalty += R_MISS_SELF_HARM

        # Missed any critical violation
        elif (gt_severity == Severity.CRITICAL
              and action.decision == ModerationDecision.APPROVE):
            penalty += R_MISS_CRITICAL

        # Over-removed clean content
        if (gt_decision == ModerationDecision.APPROVE
                and action.decision == ModerationDecision.REMOVE):
            penalty += R_OVER_REMOVE

        # Low-confidence penalty
        if action.confidence < 0.5:
            shortfall = (0.5 - action.confidence) / 0.1
            penalty += R_CONFIDENCE_PENALTY * shortfall

        # --- Assemble total reward ---
        raw_total = (
            scores["decision_score"]  * R_CORRECT_DECISION
            + scores["category_score"] * R_CORRECT_CATEGORY
            + scores["severity_score"] * R_CORRECT_SEVERITY
            + efficiency_bonus
            + consistency_bonus
            + penalty
        )
        total_reward = round(max(-1.0, min(1.0, raw_total)), 4)
        self._cumulative_reward += total_reward

        # --- Record decision ---
        self._decisions.append({
            "action": action,
            "gt_decision": gt_decision,
            "gt_category": gt_category,
            "gt_severity": gt_severity,
        })
        self._recent_decisions.append({
            "content_id": action.content_id,
            "decision":   action.decision.value,
            "category":   action.violation_category.value,
        })
        if len(self._recent_decisions) > 3:
            self._recent_decisions.pop(0)

        self._current_idx += 1

        # --- Check done ---
        if self._current_idx >= len(self._items) or self._step_number >= self.MAX_STEPS:
            self._done = True

        reward_obj = ModerationReward(
            decision_score    = scores["decision_score"]  * R_CORRECT_DECISION,
            category_score    = scores["category_score"]  * R_CORRECT_CATEGORY,
            severity_score    = scores["severity_score"]  * R_CORRECT_SEVERITY,
            efficiency_bonus  = efficiency_bonus,
            consistency_bonus = consistency_bonus,
            penalty           = penalty,
            total             = total_reward,
            explanation       = (
                f"Decision: {scores['decision_score']:.2f} | "
                f"Category: {scores['category_score']:.2f} | "
                f"Severity: {scores['severity_score']:.2f} | "
                f"Efficiency: {efficiency_bonus:.2f} | "
                f"Consistency: {consistency_bonus:.2f} | "
                f"Penalty: {penalty:.2f}"
            ),
        )

        info: Dict[str, Any] = {
            "reward_breakdown": reward_obj.dict(),
            "step": self._step_number,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "items_remaining": len(self._items) - self._current_idx,
        }

        if self._done:
            grader_fn = GRADERS[self._difficulty]
            episode_result: EpisodeResult = grader_fn(self._decisions)
            info["episode_result"] = episode_result.dict()

        obs = self._make_observation()
        return obs, total_reward, self._done, info

    # -----------------------------------------------------------------------
    # state
    # -----------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """Return the full current environment state (for debugging / logging)."""
        return {
            "env_id":            self.ENV_ID,
            "version":           self.VERSION,
            "task_id":           self._task_id,
            "difficulty":        self._difficulty,
            "step_number":       self._step_number,
            "current_idx":       self._current_idx,
            "total_items":       len(self._items),
            "done":              self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "decisions_made":    len(self._decisions),
            "violation_decision_history": self._violation_decision_history,
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _make_observation(self) -> ModerationObservation:
        """Build the current observation from internal state."""
        if self._done or self._current_idx >= len(self._items):
            # Return a terminal observation (episode over)
            last = self._items[-1] if self._items else None
            content  = last["content"]   if last else None
            user_ctx = last["user_context"] if last else None
        else:
            item_def = self._items[self._current_idx]
            content  = item_def["content"]
            user_ctx = item_def["user_context"]

        task_def = ALL_TASKS.get(self._difficulty, {})
        applicable_rules = task_def.get("applicable_rules", [])

        # Backlog pressure: rises as queue fills up
        total = len(self._items)
        remaining = total - self._current_idx
        backlog_pressure = round(remaining / total, 2) if total > 0 else 0.0

        return ModerationObservation(
            content           = content,
            user_context      = user_ctx,
            applicable_rules  = applicable_rules,
            queue_position    = self._current_idx + 1,
            queue_total       = len(self._items),
            items_processed   = self._current_idx,
            correct_so_far    = sum(
                1 for d in self._decisions
                if grade_single_action(
                    d["action"], d["gt_decision"], d["gt_category"], d["gt_severity"]
                )["decision_score"] == 1.0
            ),
            step_number       = self._step_number,
            backlog_pressure  = backlog_pressure,
            recent_decisions  = list(self._recent_decisions),
            task_id           = self._task_id or "",
            task_difficulty   = self._difficulty or "",
        )
