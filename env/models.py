"""
Typed Pydantic models for the Content Moderation OpenEnv environment.
Defines Observation, Action, and Reward schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ViolationCategory(str, Enum):
    HATE_SPEECH       = "hate_speech"
    HARASSMENT        = "harassment"
    MISINFORMATION    = "misinformation"
    SPAM              = "spam"
    EXPLICIT_CONTENT  = "explicit_content"
    VIOLENCE          = "violence"
    SELF_HARM         = "self_harm"
    NONE              = "none"


class ModerationDecision(str, Enum):
    APPROVE          = "approve"
    REMOVE           = "remove"
    ESCALATE         = "escalate"          # send to human review
    WARN_USER        = "warn_user"         # keep but flag account
    AGE_RESTRICT     = "age_restrict"      # limit audience
    SHADOW_BAN       = "shadow_ban"        # suppress without removal


class ContentType(str, Enum):
    TEXT   = "text"
    IMAGE  = "image"
    VIDEO  = "video"
    LINK   = "link"


class Severity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class UserReputation(str, Enum):
    NEW         = "new"          # < 7 days, no history
    TRUSTED     = "trusted"      # established, clean record
    FLAGGED     = "flagged"      # prior violations
    BANNED      = "banned"       # currently banned (appeal case)
    VERIFIED    = "verified"     # public figure / journalist


# ---------------------------------------------------------------------------
# Observation — what the agent sees each step
# ---------------------------------------------------------------------------

class ContentItem(BaseModel):
    """A single piece of user-generated content to moderate."""
    content_id:      str
    content_type:    ContentType
    text:            Optional[str]  = None
    image_url:       Optional[str]  = None
    platform:        str            = "social_feed"
    language:        str            = "en"
    reported_count:  int            = 0
    view_count:      int            = 0
    engagement_rate: float          = 0.0   # likes+shares / views
    has_links:       bool           = False
    metadata:        Dict[str, Any] = Field(default_factory=dict)


class UserContext(BaseModel):
    """Context about the content author."""
    user_id:          str
    reputation:       UserReputation
    account_age_days: int
    prior_violations: int = 0
    follower_count:   int = 0
    is_verified:      bool = False


class PolicyRule(BaseModel):
    """A platform policy rule that may apply."""
    rule_id:     str
    description: str
    severity:    Severity


class ModerationObservation(BaseModel):
    """Full observation returned to the agent at each step."""
    # Current item to moderate
    content:             ContentItem
    user_context:        UserContext
    applicable_rules:    List[PolicyRule]

    # Queue state
    queue_position:      int   # 1-indexed position in current task queue
    queue_total:         int   # total items in episode
    items_processed:     int
    correct_so_far:      int
    step_number:         int

    # Time pressure signal (0.0 = no pressure, 1.0 = critical backlog)
    backlog_pressure:    float = 0.0

    # Prior decisions in this episode (last 3, for context)
    recent_decisions:    List[Dict[str, str]] = Field(default_factory=list)

    # Task metadata
    task_id:             str
    task_difficulty:     str   # "easy" | "medium" | "hard"


# ---------------------------------------------------------------------------
# Action — what the agent can do
# ---------------------------------------------------------------------------

class ModerationAction(BaseModel):
    """Action submitted by the agent for a single content item."""
    content_id:          str
    decision:            ModerationDecision
    violation_category:  ViolationCategory
    severity_assessment: Severity
    confidence:          float = Field(ge=0.0, le=1.0)
    reasoning:           str   = ""   # optional chain-of-thought

    # Optional escalation details
    escalation_note:     Optional[str] = None
    policy_rule_cited:   Optional[str] = None


# ---------------------------------------------------------------------------
# Reward — structured breakdown
# ---------------------------------------------------------------------------

class ModerationReward(BaseModel):
    """Detailed reward breakdown for one moderation step."""
    decision_score:    float   # correctness of approve/remove/etc.
    category_score:    float   # correctness of violation category
    severity_score:    float   # correctness of severity
    efficiency_bonus:  float   # reward for not over-escalating
    consistency_bonus: float   # reward for consistent policy application
    penalty:           float   # deduction for bad actions

    total:             float
    explanation:       str


# ---------------------------------------------------------------------------
# Episode result (returned in info dict when done=True)
# ---------------------------------------------------------------------------

class EpisodeResult(BaseModel):
    """Summary returned when an episode ends."""
    task_id:              str
    task_difficulty:      str
    total_items:          int
    correct_decisions:    int
    accuracy:             float
    avg_reward_per_step:  float
    final_score:          float          # 0.0–1.0 for the grader
    category_breakdown:   Dict[str, Any] = Field(default_factory=dict)
    error_analysis:       List[str]      = Field(default_factory=list)
