# SBA 7(a) Loan Default Prediction 
**Advanced Topics in Machine Learning – Course Project**

This repository contains an end-to-end machine learning pipeline for **SBA 7(a) loan default prediction** (supervised learning) and **portfolio segmentation** (unsupervised learning) using the SBA 7(a) FOIA dataset (FY2020–Present) retrieved from **data.gov**.

The project is implemented in Python in a single Jupyter notebook and includes: data preparation, leakage-aware feature selection, preprocessing pipelines, model tuning and evaluation under class imbalance, final-model diagnostics, and K-Means clustering for interpretable loan segments.

---

## 1. Project Goals

### 1.1 Supervised Learning (Classification)
Predict whether a loan will default using features available **at or near approval time**.

### 1.2 Unsupervised Learning (Clustering)
Discover interpretable loan “profiles” (segments) to complement the supervised risk model with portfolio-level insights.

---

## 2. Dataset

### 2.1 Source and Collection (API / Crawling)
- **Dataset:** SBA 7(a) FOIA dataset (FY2020–Present), retrieved from **data.gov**
- **Collection method:** Downloaded as a **CSV snapshot** and loaded locally (no API calls and no web crawling).
- **Snapshot file used:**  
  `data/foia-7a-fy2020-present-asof-250930.csv`

### 2.2 Target Definition (Supervised Label)
To construct a clean, unambiguous target, only loans with **final outcomes** are used:

- **Default (1):** `LoanStatus = CHGOFF`  
- **Non-default (0):** `LoanStatus = PIF`

Other statuses (e.g., `COMMIT`, `CANCLD`, `EXEMPT`) are excluded because they may represent intermediate, cancelled, or non-comparable outcomes.

### 2.3 Leakage Prevention (Feature Selection Principle)
To avoid inflating performance through target leakage, outcome-driven fields that are only known after the loan outcome (e.g., charge-off amounts/dates, payoff dates) are excluded from the feature set. After label creation, the original `LoanStatus` field is removed from model inputs.

---

## 3. Repository Contents (Current Layout)

```
.
├── data/
│   └── foia-7a-fy2020-present-asof-250930.csv   (tracked via Git LFS)
├── SBA_project.ipynb
├── requirements.txt
└── README.md
```

**Notes**
- The dataset is large and tracked using **Git LFS**.
- All project code + narrative are contained in `SBA_project.ipynb`.

---

## 4. Methods Summary

### 4.1 Unified Preprocessing + Modeling Pipeline
All models use a consistent pipeline based on `ColumnTransformer` + `Pipeline` to ensure:
- Fair comparison across algorithms (same preprocessing)
- Reduced leakage risk during cross-validation
- Deployment-style structure (preprocessing + model packaged together)

Typical preprocessing steps:
- Missing value imputation (numeric + categorical)
- One-hot encoding for categorical features
- Scaling for distance-based methods (e.g., KNN, K-Means)

### 4.2 Models Evaluated (Supervised)
- Dummy baseline (Most Frequent)
- Logistic Regression (`class_weight="balanced"`)
- K-Nearest Neighbors (KNN)
- Decision Tree (`class_weight="balanced"`)
- Random Forest (`class_weight="balanced"`)
- Gradient Boosting (**Baseline – selected final model**)
- Gradient Boosting + SMOTE (imbalance-handling experiment)

### 4.3 Evaluation Under Class Imbalance
Because defaults are rare, evaluation emphasizes minority-class detection and ranking quality:
- **Precision / Recall / F1** (default class)
- **ROC-AUC**
- **PR-AUC (Average Precision)** (especially informative under imbalance)
- Confusion matrix and threshold-oriented diagnostics for the final model

### 4.4 Final Supervised Model
**Selected final model:** **Gradient Boosting (Baseline)**  
This model provides the best overall balance between precision and recall (highest F1 in the main comparison), while maintaining excellent ranking performance (ROC-AUC and PR-AUC).  
The SMOTE variant improves recall and slightly improves PR-AUC, but at the cost of lower precision and slightly lower F1; therefore, the baseline model is selected for robustness and simplicity.
#### Performance Highlight (Final Model)
The final **Gradient Boosting (Baseline)** model achieved **F1 = 0.833** and **Recall = 0.801** for the default class on the held-out test set, with **Precision = 0.868**. Ranking performance was excellent (**ROC-AUC = 0.979**, **PR-AUC = 0.873**), supporting risk-based screening and prioritization.

---

## 5. Unsupervised Learning (Portfolio Segmentation)

K-Means clustering is used to identify interpretable loan segments:
- A random sample is used for computational efficiency
- Clustering is performed on standardized numeric “core structure” features (e.g., approval amount, interest rate, term, jobs supported)
- The number of clusters (k) is selected using the silhouette score
- Default rates are reported **post hoc** for interpretation only (not used to create clusters)

---

## 6. How to Run

### 6.1 Clone the Repository
```bash
git clone https://github.com/rebeccaelhaj/sba-2020-2025-loan-default.git
cd sba-2020-2025-loan-default
```

### 6.2 Pull the Dataset via Git LFS
Make sure Git LFS is installed on your machine, then:
```bash
git lfs install
git lfs pull
```

### 6.3 Install Dependencies
Recommended (virtual environment):
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 6.4 Run the Notebook
```bash
jupyter notebook
```

Open:
- `SBA_project.ipynb`

**Run the notebook top-to-bottom** to reproduce the results.

---

## 7. Outputs (Produced by the Notebook)
The notebook produces (displayed as cells and plots):
- Model comparison table across all classifiers
- Confusion matrix for the final model
- ROC and PR curves
- Threshold-oriented diagnostics
- Feature-importance analysis for the final model
- K-Means silhouette analysis, cluster profiles, and clustering visualizations

---

## 8. Reproducibility
A fixed `RANDOM_STATE` is used across preprocessing, splitting, and model training. Results are reproducible by running all cells in order.

---

## 9. Limitations and Future Work
- Only loans with final outcomes (PIF/CHGOFF) are used; intermediate statuses are excluded.
- The split is random rather than time-based; temporal shifts may affect real-world performance.
- Clustering uses a compact numeric subset for interpretability; additional features and alternative clustering methods may reveal richer segmentation structure.

---

## 10. GenAI Disclosure
ChatGPT was used to assist with:
- Drafting/refining parts of the narrative text
- Writing/refactoring selected utility/organization code snippets

All preprocessing, modeling choices, experiments, and reported results were executed, reviewed, and validated by the author, who takes full responsibility for the final content.

---

## 11. Author
- **Name:** Rebecca Elhaj  
- **Course:** Advanced Topics in ML / Data Mining  
- **Institution:** Ariel University
