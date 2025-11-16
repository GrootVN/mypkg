# example_usage.py
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import shap
import pandas as pd
import numpy as np


from mypkg import SimpleClassifiers # adjust name
from mypkg import plot_tsne, plot_mulshap  # adjust name

# 1. Load a toy dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Turn X into a DataFrame (useful for SHAP)
X_df = pd.DataFrame(X, columns=feature_names)

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Create and fit the classifiers
sc = SimpleClassifiers()
sc.fit(X_train, y_train)

# 4. Predict and compute accuracies
y_pred = sc.predict(X_test)
test_acc = sc.calculate_accuracies(y_test)
print("Test accuracies:", test_acc)

# 5. Plot accuracies
fig, ax = plt.subplots(figsize=(8, 4))
sc.plot_accuracies(ax, color='tab:blue', linelabel='BreastCancer', title='Train/Test Accuracies')
plt.show()

# 6. Plot confusion matrices
sc.plot_confusion_matrices(y_test, y_pred, norm_type='true')

# 7. Compute some index sets (e.g., false negatives)
fn_indices = sc.compute_indices("fn")
print("False negative indices (per model):")
for name, idx_list in fn_indices.items():
    print(name, ":", idx_list[:10], "...")  # show first 10

# 8. t-SNE visualization (on a subset, to keep it quick)
plot_tsne(X_train.values[:500], y_train[:500], perplexities=[30, 50])

# 9. SHAP example (with one model)
#    Let's pick the first fitted model (e.g. DecisionTree or RandomForest depending on your list)
rf_model = sc.get_model(1)  # WARNING: index depends on your classifiers order

# TreeExplainer works best with tree-based models
explainer = shap.TreeExplainer(rf_model)
# Use a small subset of training data for SHAP to keep it fast
X_shap = X_train.iloc[:200]
shap_values = explainer.shap_values(X_shap)
plot_mulshap(shap_values, X_shap)
