"""
Task definitions for the Content Moderation OpenEnv environment.
Three tasks: EASY (clear-cut), MEDIUM (ambiguous context), HARD (adversarial + policy conflict).
Each task contains a dataset of content items with ground-truth labels.
"""

from __future__ import annotations

from typing import Any, Dict, List

from env.models import (
    ContentItem,
    ContentType,
    ModerationDecision,
    PolicyRule,
    Severity,
    UserContext,
    UserReputation,
    ViolationCategory,
)


# ---------------------------------------------------------------------------
# Helper dataclass (plain dict for serialisability)
# ---------------------------------------------------------------------------

def _item(
    content_id: str,
    text: str,
    content_type: ContentType,
    reported_count: int,
    view_count: int,
    engagement_rate: float,
    has_links: bool,
    gt_decision: ModerationDecision,
    gt_category: ViolationCategory,
    gt_severity: Severity,
    user_id: str,
    reputation: UserReputation,
    account_age_days: int,
    prior_violations: int = 0,
    follower_count: int = 0,
    is_verified: bool = False,
    platform: str = "social_feed",
    language: str = "en",
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "content": ContentItem(
            content_id=content_id,
            content_type=content_type,
            text=text,
            platform=platform,
            language=language,
            reported_count=reported_count,
            view_count=view_count,
            engagement_rate=engagement_rate,
            has_links=has_links,
            metadata=metadata or {},
        ),
        "user_context": UserContext(
            user_id=user_id,
            reputation=reputation,
            account_age_days=account_age_days,
            prior_violations=prior_violations,
            follower_count=follower_count,
            is_verified=is_verified,
        ),
        "gt_decision": gt_decision,
        "gt_category": gt_category,
        "gt_severity": gt_severity,
    }


# ---------------------------------------------------------------------------
# Shared policy rules
# ---------------------------------------------------------------------------

POLICY_RULES: List[PolicyRule] = [
    PolicyRule(rule_id="P001", description="No targeted hate speech based on protected characteristics.", severity=Severity.HIGH),
    PolicyRule(rule_id="P002", description="No coordinated harassment campaigns.", severity=Severity.HIGH),
    PolicyRule(rule_id="P003", description="No verifiably false health/safety misinformation.", severity=Severity.HIGH),
    PolicyRule(rule_id="P004", description="No automated spam or inauthentic behaviour.", severity=Severity.MEDIUM),
    PolicyRule(rule_id="P005", description="Explicit adult content requires age-gating.", severity=Severity.MEDIUM),
    PolicyRule(rule_id="P006", description="No graphic violence glorification.", severity=Severity.HIGH),
    PolicyRule(rule_id="P007", description="No self-harm promotion or methods.", severity=Severity.CRITICAL),
    PolicyRule(rule_id="P008", description="Satire and parody are permitted with clear framing.", severity=Severity.LOW),
    PolicyRule(rule_id="P009", description="Verified journalists may report on violence without removal.", severity=Severity.LOW),
    PolicyRule(rule_id="P010", description="Repeat violators receive escalated responses.", severity=Severity.MEDIUM),
]


# ---------------------------------------------------------------------------
# TASK 1 — EASY: Clear-cut violations, unambiguous content
# ---------------------------------------------------------------------------

EASY_TASK: Dict[str, Any] = {
    "task_id": "task_easy_001",
    "difficulty": "easy",
    "description": (
        "Moderate a queue of 10 clearly categorised content items. "
        "Violations are unambiguous. User context is straightforward. "
        "Expected accuracy for a strong agent: ≥0.90."
    ),
    "applicable_rules": [POLICY_RULES[0], POLICY_RULES[3], POLICY_RULES[5], POLICY_RULES[6]],
    "items": [
        _item(
            content_id="e001",
            text="Buy cheap followers now! Click here → spam-link.biz/promo",
            content_type=ContentType.TEXT,
            reported_count=42, view_count=500, engagement_rate=0.01, has_links=True,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.SPAM,
            gt_severity=Severity.MEDIUM,
            user_id="u001", reputation=UserReputation.NEW, account_age_days=2,
        ),
        _item(
            content_id="e002",
            text="I love hiking in the mountains! Here are my top 5 trails 🏔️",
            content_type=ContentType.TEXT,
            reported_count=0, view_count=3200, engagement_rate=0.12, has_links=False,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u002", reputation=UserReputation.TRUSTED, account_age_days=400,
        ),
        _item(
            content_id="e003",
            text="[Graphic threat] I will find you and hurt you, you piece of garbage. You don't deserve to live.",
            content_type=ContentType.TEXT,
            reported_count=18, view_count=90, engagement_rate=0.05, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.HARASSMENT,
            gt_severity=Severity.HIGH,
            user_id="u003", reputation=UserReputation.FLAGGED, account_age_days=30, prior_violations=2,
        ),
        _item(
            content_id="e004",
            text="Just finished my first marathon! 4:12 finish time, so proud of myself 🏅",
            content_type=ContentType.TEXT,
            reported_count=0, view_count=820, engagement_rate=0.18, has_links=False,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u004", reputation=UserReputation.TRUSTED, account_age_days=200,
        ),
        _item(
            content_id="e005",
            text="VACCINES CAUSE AUTISM. Do NOT inject your children with poison. Share before they delete this!!",
            content_type=ContentType.TEXT,
            reported_count=310, view_count=45000, engagement_rate=0.22, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.MISINFORMATION,
            gt_severity=Severity.HIGH,
            user_id="u005", reputation=UserReputation.FLAGGED, account_age_days=60, prior_violations=1,
        ),
        _item(
            content_id="e006",
            text="Anyone know a good recipe for banana bread? Mine keeps coming out too dense.",
            content_type=ContentType.TEXT,
            reported_count=0, view_count=150, engagement_rate=0.08, has_links=False,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u006", reputation=UserReputation.NEW, account_age_days=5,
        ),
        _item(
            content_id="e007",
            text="[Spam bot] Win $1000 Amazon gift card FREE! Limited time! DM me now! [same message posted 500 times]",
            content_type=ContentType.TEXT,
            reported_count=89, view_count=2000, engagement_rate=0.002, has_links=True,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.SPAM,
            gt_severity=Severity.MEDIUM,
            user_id="u007", reputation=UserReputation.FLAGGED, account_age_days=1, prior_violations=3,
            metadata={"duplicate_count": 500},
        ),
        _item(
            content_id="e008",
            text="Here is a detailed tutorial on how to hang yourself if life gets too hard. Step 1...",
            content_type=ContentType.TEXT,
            reported_count=24, view_count=400, engagement_rate=0.03, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.SELF_HARM,
            gt_severity=Severity.CRITICAL,
            user_id="u008", reputation=UserReputation.NEW, account_age_days=3,
        ),
        _item(
            content_id="e009",
            text="Today's weather in Chicago is 72°F and sunny — perfect day for a walk!",
            content_type=ContentType.TEXT,
            reported_count=0, view_count=320, engagement_rate=0.04, has_links=False,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u009", reputation=UserReputation.TRUSTED, account_age_days=700,
        ),
        _item(
            content_id="e010",
            text="All [ethnic group] people are criminals and should be deported. They are not human.",
            content_type=ContentType.TEXT,
            reported_count=95, view_count=1200, engagement_rate=0.08, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.HATE_SPEECH,
            gt_severity=Severity.HIGH,
            user_id="u010", reputation=UserReputation.FLAGGED, account_age_days=90, prior_violations=2,
        ),
    ],
}


# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM: Ambiguous context, policy edge-cases
# ---------------------------------------------------------------------------

MEDIUM_TASK: Dict[str, Any] = {
    "task_id": "task_medium_001",
    "difficulty": "medium",
    "description": (
        "Moderate 12 content items where context matters significantly. "
        "Satire vs. genuine hate, journalism vs. glorification, "
        "trusted user history vs. concerning content. "
        "Expected accuracy for a strong agent: ≥0.75."
    ),
    "applicable_rules": POLICY_RULES,
    "items": [
        _item(
            content_id="m001",
            text="BREAKING: Politician X photographed accepting cash from lobbyist. Unverified source but spreading fast.",
            content_type=ContentType.TEXT,
            reported_count=5, view_count=80000, engagement_rate=0.31, has_links=True,
            gt_decision=ModerationDecision.WARN_USER,
            gt_category=ViolationCategory.MISINFORMATION,
            gt_severity=Severity.MEDIUM,
            user_id="u011", reputation=UserReputation.NEW, account_age_days=4,
            metadata={"source_verified": False},
        ),
        _item(
            content_id="m002",
            text="As a nurse, I've seen COVID patients refuse the ventilator because they read online it kills you. This misinformation is literally deadly.",
            content_type=ContentType.TEXT,
            reported_count=2, view_count=5400, engagement_rate=0.14, has_links=False,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u012", reputation=UserReputation.TRUSTED, account_age_days=600,
            metadata={"profession_claimed": "nurse"},
        ),
        _item(
            content_id="m003",
            text="[Satire account, clearly labelled] BREAKING: Local man refuses to eat tide pods, cites 'not wanting to die' as reason. Experts baffled.",
            content_type=ContentType.TEXT,
            reported_count=1, view_count=12000, engagement_rate=0.28, has_links=False,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u013", reputation=UserReputation.TRUSTED, account_age_days=900,
            metadata={"account_type": "satire", "satire_label": True},
        ),
        _item(
            content_id="m004",
            text="Graphic footage of war crimes in [conflict zone]. WARNING: Disturbing content. This must be seen. — Reuters",
            content_type=ContentType.IMAGE,
            reported_count=33, view_count=220000, engagement_rate=0.19, has_links=False,
            gt_decision=ModerationDecision.AGE_RESTRICT,
            gt_category=ViolationCategory.VIOLENCE,
            gt_severity=Severity.HIGH,
            user_id="u014", reputation=UserReputation.VERIFIED, account_age_days=2000,
            is_verified=True, follower_count=890000,
            metadata={"news_org": "Reuters"},
        ),
        _item(
            content_id="m005",
            text="I can't take this anymore. Every day is the same emptiness. I don't see the point of continuing.",
            content_type=ContentType.TEXT,
            reported_count=3, view_count=80, engagement_rate=0.6, has_links=False,
            gt_decision=ModerationDecision.ESCALATE,
            gt_category=ViolationCategory.SELF_HARM,
            gt_severity=Severity.HIGH,
            user_id="u015", reputation=UserReputation.NEW, account_age_days=8,
        ),
        _item(
            content_id="m006",
            text="These politicians are all snakes who deserve to be voted out — or worse. 🐍",
            content_type=ContentType.TEXT,
            reported_count=7, view_count=4500, engagement_rate=0.09, has_links=False,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u016", reputation=UserReputation.TRUSTED, account_age_days=500,
            metadata={"hyperbole": True},
        ),
        _item(
            content_id="m007",
            text="@username you are so ugly no wonder nobody loves you. Kill yourself already, the world would be better.",
            content_type=ContentType.TEXT,
            reported_count=12, view_count=200, engagement_rate=0.07, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.HARASSMENT,
            gt_severity=Severity.HIGH,
            user_id="u017", reputation=UserReputation.FLAGGED, account_age_days=25, prior_violations=1,
        ),
        _item(
            content_id="m008",
            text="New study from Harvard: moderate alcohol consumption linked to lower heart disease risk. Link in bio.",
            content_type=ContentType.TEXT,
            reported_count=0, view_count=28000, engagement_rate=0.11, has_links=True,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u018", reputation=UserReputation.VERIFIED, account_age_days=1500, is_verified=True,
            metadata={"study_link": "harvard.edu/study/alcohol"},
        ),
        _item(
            content_id="m009",
            text="WIN WIN WIN!! Refer 5 friends and earn $50 credit. Not a scam! DM me. [account created 1 day ago]",
            content_type=ContentType.TEXT,
            reported_count=4, view_count=300, engagement_rate=0.02, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.SPAM,
            gt_severity=Severity.MEDIUM,
            user_id="u019", reputation=UserReputation.NEW, account_age_days=1,
        ),
        _item(
            content_id="m010",
            text="Historical photo: [racial slur used in 1940s protest sign]. Caption: 'What civil rights marchers faced.'",
            content_type=ContentType.IMAGE,
            reported_count=8, view_count=6700, engagement_rate=0.08, has_links=False,
            gt_decision=ModerationDecision.AGE_RESTRICT,
            gt_category=ViolationCategory.HATE_SPEECH,
            gt_severity=Severity.MEDIUM,
            user_id="u020", reputation=UserReputation.TRUSTED, account_age_days=1100,
            metadata={"educational_context": True, "historical": True},
        ),
        _item(
            content_id="m011",
            text="If you're feeling suicidal, please call 988. You are not alone. Here are some resources that helped me.",
            content_type=ContentType.TEXT,
            reported_count=0, view_count=14000, engagement_rate=0.25, has_links=True,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u021", reputation=UserReputation.TRUSTED, account_age_days=730,
            metadata={"crisis_resource": True},
        ),
        _item(
            content_id="m012",
            text="[Same exact post from 40 different new accounts in 10 minutes] 'Stock XYZ is going to moon 🚀🚀🚀'",
            content_type=ContentType.TEXT,
            reported_count=60, view_count=5000, engagement_rate=0.01, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.SPAM,
            gt_severity=Severity.HIGH,
            user_id="u022", reputation=UserReputation.NEW, account_age_days=1,
            metadata={"coordinated_accounts": 40, "time_window_minutes": 10},
        ),
    ],
}


# ---------------------------------------------------------------------------
# TASK 3 — HARD: Adversarial evasion, policy conflicts, multi-violation
# ---------------------------------------------------------------------------

HARD_TASK: Dict[str, Any] = {
    "task_id": "task_hard_001",
    "difficulty": "hard",
    "description": (
        "Moderate 15 adversarial content items designed to evade detection. "
        "Includes policy conflicts (free speech vs. safety), multi-violation posts, "
        "coded language, and appeals from banned users. "
        "Expected accuracy for a frontier model: ≥0.65."
    ),
    "applicable_rules": POLICY_RULES,
    "items": [
        _item(
            content_id="h001",
            text="Some people just 'lose oxygen' permanently, if you know what I mean 😉 #justsaying",
            content_type=ContentType.TEXT,
            reported_count=2, view_count=600, engagement_rate=0.07, has_links=False,
            gt_decision=ModerationDecision.ESCALATE,
            gt_category=ViolationCategory.VIOLENCE,
            gt_severity=Severity.HIGH,
            user_id="u023", reputation=UserReputation.FLAGGED, account_age_days=80, prior_violations=3,
            metadata={"coded_language": True},
        ),
        _item(
            content_id="h002",
            text="My novel chapter: The detective explained how the killer dissolved the body in lye to destroy evidence. 'You'd need 200L of water and...'",
            content_type=ContentType.TEXT,
            reported_count=0, view_count=1200, engagement_rate=0.09, has_links=False,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u024", reputation=UserReputation.TRUSTED, account_age_days=1200,
            metadata={"claimed_fiction": True},
        ),
        _item(
            content_id="h003",
            text="I'm not saying [politician] should be shot, I'm just saying that historically leaders who fail their people tend to meet violent ends. Food for thought.",
            content_type=ContentType.TEXT,
            reported_count=15, view_count=9000, engagement_rate=0.18, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.VIOLENCE,
            gt_severity=Severity.HIGH,
            user_id="u025", reputation=UserReputation.FLAGGED, account_age_days=45, prior_violations=2,
            metadata={"plausible_deniability": True},
        ),
        _item(
            content_id="h004",
            text="The gvt is p01son1ng the w4ter supply w1th flu0r1de to m1nd c0ntrol the p0pulation!!! SHARE SHARE SHARE",
            content_type=ContentType.TEXT,
            reported_count=6, view_count=3000, engagement_rate=0.14, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.MISINFORMATION,
            gt_severity=Severity.MEDIUM,
            user_id="u026", reputation=UserReputation.NEW, account_age_days=2,
            metadata={"character_substitution": True},
        ),
        _item(
            content_id="h005",
            text="[Verified doctor, 200k followers] Ivermectin cures COVID-19. I've treated 300 patients. Ignore the WHO.",
            content_type=ContentType.TEXT,
            reported_count=44, view_count=350000, engagement_rate=0.27, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.MISINFORMATION,
            gt_severity=Severity.HIGH,
            user_id="u027", reputation=UserReputation.VERIFIED, account_age_days=2500,
            is_verified=True, follower_count=200000,
            metadata={"expertise_claimed": "doctor", "contradicts_who": True},
        ),
        _item(
            content_id="h006",
            text="Appeal: I was banned for posting about my experience with self-harm recovery. I was sharing my story to help others heal. Please review.",
            content_type=ContentType.TEXT,
            reported_count=0, view_count=1, engagement_rate=0.0, has_links=False,
            gt_decision=ModerationDecision.ESCALATE,
            gt_category=ViolationCategory.SELF_HARM,
            gt_severity=Severity.MEDIUM,
            user_id="u028", reputation=UserReputation.BANNED, account_age_days=180, prior_violations=1,
            metadata={"is_appeal": True},
        ),
        _item(
            content_id="h007",
            text="These [slur] immigrants are stealing jobs AND bringing crime. Someone should do something. #RemoveThem",
            content_type=ContentType.TEXT,
            reported_count=28, view_count=12000, engagement_rate=0.16, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.HATE_SPEECH,
            gt_severity=Severity.HIGH,
            user_id="u029", reputation=UserReputation.FLAGGED, account_age_days=110, prior_violations=1,
        ),
        _item(
            content_id="h008",
            text="ADULTS ONLY: My new fiction story involves explicit romantic scenes between two 17-year-old characters.",
            content_type=ContentType.TEXT,
            reported_count=5, view_count=800, engagement_rate=0.04, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.EXPLICIT_CONTENT,
            gt_severity=Severity.CRITICAL,
            user_id="u030", reputation=UserReputation.NEW, account_age_days=3,
            metadata={"minors_involved": True},
        ),
        _item(
            content_id="h009",
            text="Traditional values say women belong at home. I'm not being sexist, I'm stating biological reality backed by science.",
            content_type=ContentType.TEXT,
            reported_count=9, view_count=7500, engagement_rate=0.13, has_links=False,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u031", reputation=UserReputation.TRUSTED, account_age_days=800,
            metadata={"opinion_piece": True, "no_direct_threat": True},
        ),
        _item(
            content_id="h010",
            text="[Journalist account] Undercover in the far-right rally: members openly discussed plans to intimidate minority voters at polling stations. Full report: [link]",
            content_type=ContentType.TEXT,
            reported_count=3, view_count=95000, engagement_rate=0.22, has_links=True,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u032", reputation=UserReputation.VERIFIED, account_age_days=3000,
            is_verified=True, follower_count=120000,
            metadata={"journalism": True, "news_org": "ProPublica"},
        ),
        _item(
            content_id="h011",
            text="This depression medication doesn't work. I've decided to stop taking it cold turkey and use herbs instead. Who else has done this?",
            content_type=ContentType.TEXT,
            reported_count=1, view_count=2000, engagement_rate=0.11, has_links=False,
            gt_decision=ModerationDecision.WARN_USER,
            gt_category=ViolationCategory.MISINFORMATION,
            gt_severity=Severity.HIGH,
            user_id="u033", reputation=UserReputation.NEW, account_age_days=12,
            metadata={"medical_advice": True, "dangerous_if_followed": True},
        ),
        _item(
            content_id="h012",
            text="[Bot network] 800 accounts, all created last week, all posting: 'Vote for [candidate] — he will protect YOUR children from groomers.'",
            content_type=ContentType.TEXT,
            reported_count=120, view_count=500000, engagement_rate=0.03, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.SPAM,
            gt_severity=Severity.CRITICAL,
            user_id="u034", reputation=UserReputation.NEW, account_age_days=3,
            metadata={"coordinated_accounts": 800, "election_interference": True},
        ),
        _item(
            content_id="h013",
            text="To all my followers going through hard times: I was at my lowest point last year. Here's what pulled me through (thread 1/8)",
            content_type=ContentType.TEXT,
            reported_count=0, view_count=45000, engagement_rate=0.32, has_links=False,
            gt_decision=ModerationDecision.APPROVE,
            gt_category=ViolationCategory.NONE,
            gt_severity=Severity.LOW,
            user_id="u035", reputation=UserReputation.TRUSTED, account_age_days=1400, follower_count=50000,
        ),
        _item(
            content_id="h014",
            text="New research: 5G towers cause cancer. Scientists are being silenced. Here's the proof [link to personal blog]",
            content_type=ContentType.TEXT,
            reported_count=18, view_count=22000, engagement_rate=0.17, has_links=True,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.MISINFORMATION,
            gt_severity=Severity.HIGH,
            user_id="u036", reputation=UserReputation.FLAGGED, account_age_days=55, prior_violations=1,
            metadata={"source": "personal_blog", "contradicts_consensus": True},
        ),
        _item(
            content_id="h015",
            text="Looking for online community: struggling with eating less than 500 cal/day, need tips to push through hunger. #fasting #wellness",
            content_type=ContentType.TEXT,
            reported_count=2, view_count=700, engagement_rate=0.08, has_links=False,
            gt_decision=ModerationDecision.REMOVE,
            gt_category=ViolationCategory.SELF_HARM,
            gt_severity=Severity.HIGH,
            user_id="u037", reputation=UserReputation.NEW, account_age_days=6,
            metadata={"eating_disorder_signals": True, "disguised_as_wellness": True},
        ),
    ],
}


ALL_TASKS: Dict[str, Dict] = {
    "easy":   EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard":   HARD_TASK,
}
