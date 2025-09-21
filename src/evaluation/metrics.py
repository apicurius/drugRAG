#!/usr/bin/env python3
"""
Binary Classification Metrics Module for DrugRAG Evaluation

This module implements the 5 binary classification metrics reported in the manuscript:
1. Accuracy - (TP + TN) / (TP + TN + FP + FN)
2. F1 Score - 2 × (Precision × Sensitivity) / (Precision + Sensitivity)
3. Precision - TP / (TP + FP)
4. Sensitivity (Recall) - TP / (TP + FN)
5. Specificity - TN / (TN + FP)

These align with the formulas defined in the revised manuscript.
"""

from typing import List, Tuple, Union, Dict, Any
import numpy as np


def normalize_prediction(prediction: Union[str, int, float]) -> int:
    """
    Normalize predictions to binary 0/1 format.

    Args:
        prediction: Can be 'YES'/'NO', 1/0, True/False, numeric, or 'UNKNOWN'

    Returns:
        int: 0 or 1 (UNKNOWN treated as 0)
    """
    if isinstance(prediction, str):
        prediction = prediction.strip().upper()
        if prediction.startswith('YES'):
            return 1
        elif prediction.startswith('NO') or prediction == 'UNKNOWN' or prediction == 'ERROR':
            return 0
        else:
            # Try to convert string to number
            try:
                return int(float(prediction))
            except ValueError:
                # Default UNKNOWN/unparseable predictions to 0 (NO)
                return 0

    elif isinstance(prediction, (int, float, bool)):
        return int(bool(prediction))

    else:
        # Default any unsupported type to 0
        return 0


def calculate_confusion_matrix(y_true: List[Union[str, int, float]],
                             y_pred: List[Union[str, int, float]]) -> Tuple[int, int, int, int]:
    """
    Calculate confusion matrix components: TP, TN, FP, FN

    Args:
        y_true: Ground truth labels (1 for positive, 0 for negative)
        y_pred: Predicted labels (1 for positive, 0 for negative)

    Returns:
        Tuple[int, int, int, int]: (TP, TN, FP, FN)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    # Normalize all predictions to 0/1
    y_true_norm = [normalize_prediction(y) for y in y_true]
    y_pred_norm = [normalize_prediction(y) for y in y_pred]

    tp = sum(1 for true, pred in zip(y_true_norm, y_pred_norm) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true_norm, y_pred_norm) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true_norm, y_pred_norm) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true_norm, y_pred_norm) if true == 1 and pred == 0)

    return tp, tn, fp, fn


def calculate_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    """
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    return (tp + tn) / total


def calculate_precision(tp: int, fp: int) -> float:
    """
    Calculate precision: TP / (TP + FP)
    """
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calculate_sensitivity(tp: int, fn: int) -> float:
    """
    Calculate sensitivity (recall): TP / (TP + FN)
    """
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def calculate_specificity(tn: int, fp: int) -> float:
    """
    Calculate specificity: TN / (TN + FP)
    """
    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)


def calculate_f1_score(precision: float, sensitivity: float) -> float:
    """
    Calculate F1 score: 2 × (Precision × Sensitivity) / (Precision + Sensitivity)
    """
    if precision + sensitivity == 0:
        return 0.0
    return 2 * (precision * sensitivity) / (precision + sensitivity)


def calculate_binary_classification_metrics(y_true: List[Union[str, int, float]],
                                          y_pred: List[Union[str, int, float]]) -> Dict[str, float]:
    """
    Calculate all 5 binary classification metrics reported in the manuscript.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dict containing all metrics: accuracy, precision, sensitivity, specificity, f1_score
    """
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)

    accuracy = calculate_accuracy(tp, tn, fp, fn)
    precision = calculate_precision(tp, fp)
    sensitivity = calculate_sensitivity(tp, fn)
    specificity = calculate_specificity(tn, fp)
    f1_score = calculate_f1_score(precision, sensitivity)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1_score,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def print_metrics_summary(metrics: Dict[str, float], title: str = "Binary Classification Metrics") -> None:
    """
    Print a formatted summary of the metrics matching manuscript format.

    Args:
        metrics: Dictionary returned by calculate_binary_classification_metrics
        title: Title for the summary
    """
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"TP: {metrics['tp']}, TN: {metrics['tn']}")
    print(f"FP: {metrics['fp']}, FN: {metrics['fn']}")


def validate_metrics_against_manuscript() -> None:
    """
    Validate our implementation against the expected values from the manuscript.
    This can be used for testing.
    """
    # Test case: Perfect classification
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0]

    metrics = calculate_binary_classification_metrics(y_true, y_pred)
    assert abs(metrics['accuracy'] - 1.0) < 1e-6
    assert abs(metrics['precision'] - 1.0) < 1e-6
    assert abs(metrics['sensitivity'] - 1.0) < 1e-6
    assert abs(metrics['specificity'] - 1.0) < 1e-6
    assert abs(metrics['f1_score'] - 1.0) < 1e-6

    # Test case with YES/NO strings
    y_true_str = ['YES', 'YES', 'NO', 'NO']
    y_pred_str = ['YES', 'NO', 'NO', 'NO']

    metrics_str = calculate_binary_classification_metrics(y_true_str, y_pred_str)
    # TP=1, TN=2, FP=0, FN=1
    # Accuracy = (1+2)/(1+2+0+1) = 3/4 = 0.75
    assert abs(metrics_str['accuracy'] - 0.75) < 1e-6

    print("✅ All validation tests passed!")


if __name__ == "__main__":
    # Run validation when executed directly
    validate_metrics_against_manuscript()

    # Example usage
    y_true = ['YES', 'YES', 'NO', 'NO', 'YES', 'NO']
    y_pred = ['YES', 'NO', 'NO', 'NO', 'YES', 'YES']

    metrics = calculate_binary_classification_metrics(y_true, y_pred)
    print_metrics_summary(metrics, "Example DrugRAG Evaluation")