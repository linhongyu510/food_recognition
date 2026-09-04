"""Tests for metrics, verified against hand-computed values."""

from __future__ import annotations

import pytest
import torch

from food_recognition.metrics import compute_metrics, confusion_matrix, topk_accuracy


def test_confusion_matrix_is_true_by_pred():
    targets = torch.tensor([0, 0, 1, 1, 2])
    preds = torch.tensor([0, 1, 1, 1, 2])
    matrix = confusion_matrix(targets, preds, num_classes=3)

    # row = true label, column = prediction
    assert matrix.tolist() == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]
    assert int(matrix.sum()) == 5


def test_confusion_matrix_rejects_out_of_range_labels():
    with pytest.raises(ValueError, match="outside"):
        confusion_matrix(torch.tensor([0, 3]), torch.tensor([0, 0]), num_classes=3)


def test_confusion_matrix_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        confusion_matrix(torch.tensor([0, 1]), torch.tensor([0]), num_classes=2)


def test_metrics_match_hand_computed_values():
    # class 0: 1 correct of 2 true, predicted once  -> P=1.0,  R=0.5
    # class 1: 2 correct of 2 true, predicted 3x    -> P=2/3,  R=1.0
    # class 2: 1 correct of 1 true, predicted once  -> P=1.0,  R=1.0
    targets = torch.tensor([0, 0, 1, 1, 2])
    preds = torch.tensor([0, 1, 1, 1, 2])
    report = compute_metrics(targets, preds, num_classes=3)

    assert report.accuracy == pytest.approx(4 / 5)

    assert report.per_class_precision[0] == pytest.approx(1.0)
    assert report.per_class_recall[0] == pytest.approx(0.5)
    assert report.per_class_f1[0] == pytest.approx(2 * 1.0 * 0.5 / 1.5)

    assert report.per_class_precision[1] == pytest.approx(2 / 3)
    assert report.per_class_recall[1] == pytest.approx(1.0)

    expected_macro_p = (1.0 + 2 / 3 + 1.0) / 3
    assert report.macro_precision == pytest.approx(expected_macro_p)

    expected_weighted_r = (0.5 * 2 + 1.0 * 2 + 1.0 * 1) / 5
    assert report.weighted_recall == pytest.approx(expected_weighted_r)

    assert report.support == [2, 2, 1]


def test_perfect_predictions_give_all_ones():
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    report = compute_metrics(targets, targets.clone(), num_classes=3)

    assert report.accuracy == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.weighted_f1 == pytest.approx(1.0)


def test_never_predicted_class_scores_zero_not_nan():
    # class 2 is never predicted and never true
    targets = torch.tensor([0, 0, 1, 1])
    preds = torch.tensor([0, 0, 1, 1])
    report = compute_metrics(targets, preds, num_classes=3)

    assert report.per_class_precision[2] == 0.0
    assert report.per_class_f1[2] == 0.0
    # macro average ignores the absent class rather than being dragged to 2/3
    assert report.macro_f1 == pytest.approx(1.0)


def test_single_label_micro_average_equals_accuracy():
    targets = torch.tensor([0, 1, 2, 2, 1])
    preds = torch.tensor([0, 2, 2, 2, 1])
    report = compute_metrics(targets, preds, num_classes=3)

    matrix = torch.tensor(report.matrix)
    micro_precision = matrix.diag().sum().item() / matrix.sum().item()
    assert report.accuracy == pytest.approx(micro_precision)


def test_report_dict_and_table_render():
    targets = torch.tensor([0, 1, 1])
    preds = torch.tensor([0, 1, 0])
    report = compute_metrics(
        targets, preds, num_classes=2, class_names=["bread", "soup"], loss=0.25
    )

    data = report.to_dict()
    assert data["loss"] == pytest.approx(0.25)
    assert [row["name"] for row in data["per_class"]] == ["bread", "soup"]
    assert data["confusion_matrix"] == [[1, 0], [1, 1]]

    table = report.format_table()
    assert "bread" in table and "soup" in table
    assert "weighted avg" in table


def test_topk_accuracy():
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.0],  # true 1 -> top1 correct
            [0.7, 0.2, 0.1],  # true 1 -> in top2
            [0.5, 0.3, 0.2],  # true 2 -> only in top3
        ]
    )
    targets = torch.tensor([1, 1, 2])

    assert topk_accuracy(logits, targets, k=1) == pytest.approx(1 / 3)
    assert topk_accuracy(logits, targets, k=2) == pytest.approx(2 / 3)
    assert topk_accuracy(logits, targets, k=3) == pytest.approx(1.0)
    # k larger than the class count is clamped, not an error
    assert topk_accuracy(logits, targets, k=99) == pytest.approx(1.0)
