# SBA 7(a) Loan Default Prediction + Portfolio Segmentation (Machine Learning II Course Project)

This repository contains the complete course project for supervised **loan default prediction** and unsupervised **portfolio segmentation** using the SBA 7(a) FOIA dataset (FY2020–Present) retrieved from **data.gov**.

The project implements an end-to-end machine learning workflow in Python, including data preparation, feature engineering, model training and tuning, evaluation under class imbalance, model interpretation, and K-Means clustering for segment discovery.

---

## 1. Project Overview

### Objectives
1. **Supervised Learning (Classification)**  
   Predict whether a loan will default using features available at or near approval time.

2. **Unsupervised Learning (Clustering)**  
   Identify interpretable loan “profiles” (segments) to complement the supervised results with portfolio-level insights.

---

## 2. Dataset (data.gov)

- **Dataset:** SBA 7(a) FOIA dataset (FY2020–Present)  
- **Snapshot file used in this project:** `foia-7a-fy2020-present-asof-250930.csv`

### Target Definition (Supervised)
The supervised dataset uses only loans with final outcomes to ensure an unambiguous label:

- **Default (1):** `LoanStatus = CHGOFF`  
- **Non-default (0):** `LoanStatus = PIF`

Other statuses (e.g., `COMMIT`, `CANCLD`, `EXEMPT`) are excluded because they may represent intermediate, incomplete, or non-comparable outcomes.

### Target Leakage Prevention
Outcome-driven fields that are only known after the outcome (e.g., charge-off amounts or outcome dates) are excluded from the feature set to prevent target leakage. After defining the target label, the original `LoanStatus` column is removed from the model inputs.

---

## 3. Repository Structure

A recommended structure (adapt if your repo differs):

```
.
├── notebooks/
│   └── SBA_project_2.ipynb
├── data/
│   └── foia-7a-fy2020-present-asof-250930.csv   (may be excluded from Git if too large)
├── figures/
│   └── feature_importance_baseline.png
├── requirements.txt
└── README.md
```

**Notes**
- If the dataset is too large for GitHub, keep it locally under `data/` and add it to `.gitignore`.
- Saving figures to `figures/` keeps the notebook cleaner and supports reproducibility.

---

## 4. Methods Summary

### 4.1 Supervised Learning Pipeline
The supervised workflow follows a standard ML pipeline:

1. Data loading and cleaning  
2. Target construction (`CHGOFF` vs `PIF`)  
3. Train/test split with stratification (preserves the default rate)  
4. Preprocessing with `ColumnTransformer`:
   - Missing value imputation  
   - Scaling for numeric variables  
   - One-hot encoding for categorical variables  
5. Model tuning via `GridSearchCV` (3-fold CV), optimizing **F1** for the default class  
6. Final evaluation on the held-out test set using imbalance-aware metrics

### 4.2 Models Evaluated
Multiple classifiers were evaluated under class imbalance:

- Dummy baseline (Most Frequent)
- Logistic Regression (`class_weight="balanced"`)
- K-Nearest Neighbors (KNN)
- Decision Tree (`class_weight="balanced"`)
- Random Forest (`class_weight="balanced"`)
- Gradient Boosting (**Baseline**)
- Gradient Boosting + SMOTE (imbalance-handling experiment)

### 4.3 Evaluation Metrics
Because defaults are rare, evaluation emphasizes minority-class performance:

- **Precision, Recall, F1** (for the default class)
- **ROC-AUC**
- **PR-AUC (Average Precision)** (more informative under imbalance)
- Confusion matrix for the final model
- Threshold selection using CV-train out-of-fold probabilities (to avoid tuning on the test set)

### 4.4 Final Supervised Model
**Final model selected:** **Gradient Boosting (Baseline)**

This model achieved the strongest overall balance on the test set (highest F1 and strong precision) while maintaining excellent ranking performance (ROC-AUC and PR-AUC).  
The SMOTE variant improved recall and slightly improved PR-AUC, but at the cost of lower precision and slightly lower F1. For robustness and simplicity, the baseline Gradient Boosting model is selected as the final supervised approach.

---

## 5. Unsupervised Learning (Clustering)

To complement supervised prediction with portfolio-level insights, K-Means clustering is applied:

- A random sample of 20,000 loans is used for computational efficiency.
- Clustering is performed on standardized numeric features:
  - `GrossApproval`, `InitialInterestRate`, `TerminMonths`, `JobsSupported`
- The number of clusters (k) is selected using the silhouette score.
- Cluster profiles are summarized, and the observed default rate is reported **post hoc** for interpretation only (it is not used to form clusters).

Clustering provides interpretable segments that help understand portfolio structure (e.g., small high-rate loans vs larger/longer-term structures), supporting differentiated risk monitoring.

---

## 6. How to Run

### 6.1 Install Dependencies
It is recommended to use a virtual environment, then install requirements:

```bash
pip install -r requirements.txt
```

### 6.2 Run the Notebook
Launch Jupyter and run the notebook top-to-bottom:

```bash
jupyter notebook
```

Open:

- `notebooks/SBA_project_2.ipynb`

#### Reproducibility
- The notebook uses a fixed `RANDOM_STATE`.
- Results are reproducible by running all cells in order.

---

## 7. Outputs

The notebook produces:
- Model comparison table across all classifiers
- Confusion matrix for the final model
- ROC curve and PR curve
- Threshold selection analysis
- Feature importance visualization
- K-Means silhouette plot, cluster profile table, and cluster visualizations (PCA and boxplots)

---

## 8. Limitations and Future Work

- Only loans with final outcomes (PIF/CHGOFF) are used for supervised learning; intermediate statuses are excluded.
- The split is random rather than time-based; temporal shifts may affect real-world performance.
- Clustering uses a limited numeric feature subset for interpretability; additional features and alternative clustering methods may reveal richer structure.

---

## 9. GenAI Disclosure

<span style="color:red"><b>GenAI-assisted work:</b></span> ChatGPT was used to assist with **drafting parts of the narrative text** and **writing/refactoring selected code snippets** (e.g., utilities, organization, and phrasing). All preprocessing, modeling choices, experiments, and reported results were **executed, verified, and validated by the author**, who takes full responsibility for the final content.

---

## 10. Author

- **Name:** Your Name  
- **Course:** Machine Learning II (Advanced Topics in ML / Data Mining)  
- **Institution:** Ariel University (update if needed)
