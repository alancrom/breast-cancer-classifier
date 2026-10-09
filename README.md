# Breast Cancer Classifier

**Logistic regression pipelines that classify breast tumors as malignant or benign, built to miss as few malignant cases as possible.**

In a diagnostic setting, the costly error is a false negative: calling a malignant tumor benign. This project compares three preprocessing strategies for a logistic regression model and selects the one with the best recall on the malignant class.

📓 **[Open the notebook](breast_cancer_logistic_regression.ipynb)** (outputs included; text in Spanish)

---

## Results

Test set: 114 tumors (20% hold-out).

| Model | Preprocessing | Recall | Precision | Accuracy |
|---|---|---|---|---|
| `Correlation_Clean` | Drop highly correlated features | 0.762 | 0.889 | 0.877 |
| `Standard_PCA` | Standard scaling + PCA (7 components, 91% of variance) | 0.929 | 1.000 | 0.974 |
| **`Correlation_Yeo`** | Drop correlated features + Yeo-Johnson transform | **0.952** | **1.000** | **0.982** |

Best model in detail:

- **ROC AUC: 0.992**
- **Confusion matrix:** 72 true negatives, 0 false positives, 2 false negatives, 40 true positives

## Approach

1. **Exploration.** 569 tumors, 30 numeric features, no missing values or duplicates. Classes are moderately imbalanced (62.7% benign, 37.3% malignant), so accuracy alone would be misleading.
2. **Redundancy reduction.** Feature pairs with correlation above 0.9 were identified and 15 columns were dropped: every `_worst` column, the `perimeter` and `area` measures (both functions of `radius`), and `concavity_mean`. The final model uses 15 predictors.
3. **Modeling.** Three scikit-learn pipelines, each ending in logistic regression, so that scaling, PCA, and power transforms are fitted on the training data only.
4. **Evaluation.** Recall, precision, and accuracy for every model, plus the confusion matrix, ROC curve, and predicted-probability distribution for the best one.

## Key takeaways

- Adding a Yeo-Johnson power transform, which corrects skewness and standardizes the features, raised recall from 0.762 to 0.952 on the same 15 features.
- The most influential features are `concave points_mean`, `texture_mean`, and `radius_mean`, all pushing the prediction toward malignant.
- The notebook closes with a discussion of the decision threshold: lowering it below the default 0.5 would raise recall further at the cost of more false positives, a reasonable trade in cancer screening.

## Limitations

- Results come from a single 80/20 train/test split (`random_state=1`); no cross-validation was run.
- The test set is small (114 samples, 42 of them malignant), so each missed tumor moves recall by about 2.4 points.
- The list of correlated features to drop was chosen on the full dataset, before the train/test split.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook breast_cancer_logistic_regression.ipynb
```

The notebook expects the dataset at `data/breast_cancer.csv`, with an `id` column, a `diagnosis` column (`B`/`M`), and the 30 feature columns.

## Data

[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), from the UCI Machine Learning Repository.

## Tools

Python · pandas · NumPy · scikit-learn · Matplotlib · seaborn

## Context and credits

- Academic team project (4 members), completed during the M.Sc. in Applied Artificial Intelligence at Tecnológico de Monterrey.
- **My role:** TODO — one or two lines on what you personally did.
- **AI-use disclosure:** as stated at the end of the notebook, Gemini was used for code optimization and debugging.
