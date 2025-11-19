# example_usage.py
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import shap
import pandas as pd
import numpy as np

from mypkg import SimpleClassifiers  # adjust name if needed
from mypkg import plot_tsne, plot_mulshap  # adjust name if needed

# ============================================================
# 1. Load a bc dataset
# ============================================================
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Turn X into a DataFrame (useful for SHAP and plotting)
X_df = pd.DataFrame(X, columns=feature_names)

# ============================================================
# 2. Train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.3, random_state=42, stratify=y
)

# ============================================================
# 3. Create and fit the classifiers (SimpleClassifiers wrapper)
# ============================================================
sc = SimpleClassifiers()
sc.fit(X_train, y_train)

# ============================================================
# 4. Predict and compute accuracies on the hold-out test set
# ============================================================
y_pred = sc.predict(X_test)
test_acc = sc.calculate_accuracies(y_test)
print("Test accuracies (hold-out split):")
for name, acc in test_acc.items():
    print(f"  {name}: {acc:.4f}")

# ============================================================
# 5. Plot train/test accuracies
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
sc.plot_accuracies(
    ax,
    color="tab:blue",
    linelabel="BreastCancer",
    title="Train/Test Accuracies (Hold-out Split)",
)
plt.show()

# ============================================================
# 6. Plot confusion matrices (on hold-out test set)
# ============================================================
sc.plot_confusion_matrices(y_pred, y_test, norm_type="true")

# ============================================================
# 7. Compute some index sets (e.g., false negatives on training set)
# ============================================================
fn_indices = sc.compute_indices("fn")
print("\nFalse negative indices in training (per model):")
for name, idx_list in fn_indices.items():
    print(name, ":", idx_list[:10], "...")  # show first 10

# ============================================================
# 8. t-SNE visualization (on a subset, to keep it quick)
# ============================================================
plot_tsne(X_train.values[:500], y_train[:500], perplexities=[30, 50, 20, 70, 10], saveimg="tsne_breastcancer.png")

# ============================================================
# 9. SHAP example (with one model)
#    Let's pick the second fitted model (index depends on your classifier list)
# ============================================================
rf_model = sc.get_model(1)  # WARNING: index depends on the order in SimpleClassifiers

# TreeExplainer works best with tree-based models
explainer = shap.TreeExplainer(rf_model)

# Use a small subset of training data for SHAP to keep it fast
X_shap = X_train.iloc[:200]
shap_values = explainer.shap_values(X_shap)
plot_mulshap(shap_values, X_shap)

# ============================================================
# 10. CROSS-VALIDATION SECTION (optional, on full dataset)
# ============================================================
# This uses the SimpleClassifiers.cross_validate() method.
# It runs k-fold CV for EACH model inside SimpleClassifiers using the full dataset.
# Note: this is independent of the earlier train/test split.

from sklearn.model_selection import StratifiedKFold

# Define a CV splitter (stratified for classification problems)
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run cross-validation for all models
cv_results = sc.cross_validate(
    X_df,
    y,
    cv=cv_splitter,
    scoring=["accuracy", "f1"],
    n_jobs=-1,
    return_estimator=False,
)

import numpy as np

print("\nCross-validation results (5-fold):")
print("{:<30} {:>22} {:>22}".format("Model", "Accuracy (mean ± std)", "F1-score (mean ± std)"))
print("-" * 80)

for name, res in cv_results.items():
    acc_mean = np.mean(res["test_accuracy"])
    acc_std  = np.std(res["test_accuracy"])
    f1_mean  = np.mean(res["test_f1"])
    f1_std   = np.std(res["test_f1"])

    print("{:<30} {:>11.4f} ± {:<8.4f} {:>11.4f} ± {:<8.4f}".format(
        name,
        acc_mean, acc_std,
        f1_mean, f1_std
    ))
