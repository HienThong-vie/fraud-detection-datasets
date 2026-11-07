"""Map model scores to review decisions."""

from __future__ import annotations


def make_review_decision(score: float, threshold: float) -> dict:
    high_priority_threshold = max(0.80, threshold)
    if score >= high_priority_threshold:
        label = "HIGH_PRIORITY_REVIEW"
        explanation = "The model score is above the high-priority review boundary."
    elif score >= threshold:
        label = "NEEDS_REVIEW"
        explanation = "The model score is at or above the frozen manual-review threshold."
    else:
        label = "BELOW_REVIEW_THRESHOLD"
        explanation = "The model score is below the frozen manual-review threshold."
    return {
        "label": label,
        "score": float(score),
        "threshold": float(threshold),
        "high_priority_threshold": float(high_priority_threshold),
        "decision_source": "trained ML classifier",
        "explanation": explanation,
        "human_review_required": label != "BELOW_REVIEW_THRESHOLD",
    }
