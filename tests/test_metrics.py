from handbook_fraud.metrics import best_f1_threshold, binary_metrics


def test_metrics_for_separable_scores() -> None:
    target = [0, 0, 1, 1]
    probability = [0.1, 0.2, 0.8, 0.9]
    threshold = best_f1_threshold(target, probability)
    metrics = binary_metrics(target, probability, threshold)
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0
