# Model Performance Evaluation Report

## Confusion Matrix
| | Predicted Normal | Predicted Fraud |
|---|---|---|
| Actual Normal | 12119 | 34 |
| Actual Fraud | 119 | 128 |

## Classification Metrics (thresholded at 0.5)
- Precision: 0.790
- Recall: 0.518
- F1-score: 0.626

## Threshold-independent Metrics
- ROC-AUC: 0.758
- PR-AUC: 0.419 (Baseline ≈ 0.020)

## Detailed Metrics from sklearn
{'0': {'precision': 0.9902761889197581, 'recall': 0.9972023368715544, 'f1-score': 0.9937271944569719, 'support': 12153.0}, '1': {'precision': 0.7901234567901234, 'recall': 0.5182186234817814, 'f1-score': 0.6259168704156479, 'support': 247.0}, 'accuracy': 0.9876612903225807, 'macro avg': {'precision': 0.8901998228549408, 'recall': 0.7577104801766679, 'f1-score': 0.8098220324363099, 'support': 12400.0}, 'weighted avg': {'precision': 0.9862892756265307, 'recall': 0.9876612903225807, 'f1-score': 0.986400650099052, 'support': 12400.0}}

## Interpretation
The model demonstrates strong ranking ability (ROC-AUC=0.758) and a trade-off between recall (0.518) and precision (0.790). Missed fraud cases (FN=119) may result in revenue loss, while false positives (FP=34) increase inspection costs.
