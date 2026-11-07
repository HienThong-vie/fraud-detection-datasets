"""Calibration-only review policies; confidence bounds do not guarantee future-period FPR."""
import numpy as np
from priority_search import fpr_threshold

BUDGETS = (.005, .01, .02, .03, .05)
CAPACITIES = (None, .005, .01, .02, .05)
POLICIES = ('point', 'ucb95')


def policy_key(policy, budget, capacity=None):
    """Dictionary key of one policy, for example 'point|0.05|none'."""
    return f'{policy}|{float(budget)}|{float(capacity) if capacity is not None else "none"}'


def wilson_upper(errors, n, z=1.6448536269514722):
    """One-sided 95% Wilson upper bound of an error rate (errors out of n); used by the conservative ucb95
    policy."""
    p = errors / n
    return float((p + z * z / (2 * n) + z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
                 / (1 + z * z / n))


def count_threshold(scores, permitted):
    """Smallest cutoff that lets at most `permitted` of the given scores reach it (score >= cutoff means
    flagged)."""
    ordered = np.sort(np.asarray(scores, dtype='float64'))
    if not len(ordered):
        raise ValueError('Calibration scores are required')
    if permitted >= len(ordered):
        return float(np.nextafter(ordered[0], -np.inf))
    return float(np.nextafter(ordered[len(ordered) - permitted - 1], np.inf))


def calibrate(y, scores):
    """Calibrate review thresholds on the calibration month for every FPR budget, policy (point or ucb95) and
    optional review capacity."""
    y, scores = np.asarray(y), np.asarray(scores, dtype='float64')
    negatives = scores[y == 0]
    result = {}
    for budget in BUDGETS:
        point = fpr_threshold(y, scores, budget)
        low, high = 0, int(np.floor(budget * len(negatives)))
        while low < high:
            middle = (low + high + 1) // 2
            if wilson_upper(middle, len(negatives)) <= budget:
                low = middle
            else:
                high = middle - 1
        conservative = count_threshold(negatives, low)
        for policy, threshold in [('point', point), ('ucb95', conservative)]:
            for capacity in CAPACITIES:
                cut = threshold if capacity is None else max(threshold, count_threshold(
                    scores, int(np.floor(capacity * len(scores)))))
                errors = int(np.sum(negatives >= cut))
                alert_rate = float(np.mean(scores >= cut))
                upper = wilson_upper(errors, len(negatives))
                feasible = errors / len(negatives) <= budget + 1e-12
                if policy == 'ucb95':
                    feasible = feasible and upper <= budget + 1e-12
                if capacity is not None:
                    feasible = feasible and alert_rate <= capacity + 1e-12
                result[policy_key(policy, budget, capacity)] = dict(
                    policy=policy, fpr_budget=budget, review_capacity=capacity, threshold=cut,
                    calibration_fpr=errors / len(negatives), calibration_fpr_upper95=upper,
                    calibration_alert_rate=alert_rate, calibration_feasible=bool(feasible))
    return result
