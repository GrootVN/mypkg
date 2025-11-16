import matplotlib.pyplot as plt
import numpy as np
import shap
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches



def plot_mulshap(shap_values, X):
    """
    Plot multi-class SHAP summaries and print top features.

    Parameters
    ----------
    shap_values : list or np.ndarray
        - For tree-based multi-class models, this is usually a list of length n_classes,
          each element an array of shape (n_samples, n_features).
        - It can also be:
            * a single 2D array: (n_samples, n_features)
            * a 3D array: (n_samples, n_features, n_classes)
    X : pandas.DataFrame
        Feature matrix used to compute SHAP values (columns = feature names).

    Notes
    -----
    - Computes mean |SHAP| per feature across classes and samples.
    - Prints top 10 features.
    - Plots a SHAP summary plot per class (or a single one for 2D case).
    """
    print("X shape:", X.shape)
    print("shap_values raw type:", type(shap_values))

    arr = np.array(shap_values)
    print("shap_values array shape:", arr.shape)

    # ----- 1) Compute mean absolute SHAP importance per feature -----

    if isinstance(shap_values, list):
        # Typical tree multi-class case:
        # shap_values is a list: [ (n_samples, n_features), ..., per class ]
        # arr shape: (n_classes, n_samples, n_features)
        # Mean over classes (axis=0) and samples (axis=1) => (n_features,)
        mean_abs_shap = np.mean(np.abs(arr), axis=(0, 1))
    elif arr.ndim == 3:
        # Assume shape: (n_samples, n_features, n_classes)
        # Mean over samples and classes => (n_features,)
        mean_abs_shap = np.mean(np.abs(arr), axis=(0, 2))
    elif arr.ndim == 2:
        # Shape: (n_samples, n_features)
        mean_abs_shap = np.mean(np.abs(arr), axis=0)
    else:
        raise ValueError(f"Unexpected shap_values shape: {arr.shape}")

    if len(mean_abs_shap) != X.shape[1]:
        print("WARNING: feature dimension mismatch between SHAP values and X.")
        print("len(mean_abs_shap) =", len(mean_abs_shap), "| n_features =", X.shape[1])

    # ----- 2) Build importance DataFrame and print top features -----

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_abs_shap
    })

    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    top_10_features = feature_importance.head(10)
    print("Top 10 most significant features:")
    print(top_10_features)

    # ----- 3) SHAP summary plots -----

    if isinstance(shap_values, list):
        # One summary plot per class
        for class_index, class_sv in enumerate(shap_values):
            print(f"Summary plot for class {class_index}")
            shap.summary_plot(class_sv, features=X)
    elif arr.ndim == 3:
        # 3D array: (n_samples, n_features, n_classes)
        n_classes = arr.shape[2]
        for class_index in range(n_classes):
            print(f"Summary plot for class {class_index}")
            shap.summary_plot(arr[..., class_index], features=X)
    else:
        # 2D array: single-output case
        print("Summary plot (single output)")
        shap.summary_plot(arr, features=X)



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne(X, y, perplexities=[5, 30, 50, 70, 100], saveimg=None):
    """
    Plot t-SNE visualizations with different perplexities and a legend for class labels.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Class labels (integer or string).
    perplexities : list of float, optional
        List of perplexity values to use for t-SNE.
    saveimg : str or None, optional
        If not None, path to save the resulting figure (e.g., 'tsne.png').
        If None, the figure is only displayed.

    Notes
    -----
    - Creates one subplot per perplexity.
    - Colors points by class label, with a legend on the right.
    - Uses sklearn.manifold.TSNE with n_components=2.
    """
    fig, axes = plt.subplots(1, len(perplexities), figsize=(5 * len(perplexities), 5))
    
    # If only one perplexity, axes may not be an array
    if len(perplexities) == 1:
        axes = [axes]
    
    # Get unique class labels and assign colors
    unique_labels = np.unique(y)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: color for label, color in zip(unique_labels, colors)}
    
    # Plot t-SNE for each perplexity
    for i, perplexity in enumerate(perplexities):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        # Scatter plot with mapped colors
        axes[i].scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=[label_to_color[label] for label in y],
            cmap='viridis',
            s=5
        )
        
        axes[i].set_title(f't-SNE (perplexity={perplexity})')
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')
    
    # Create a legend
    legend_patches = [
        mpatches.Patch(color=label_to_color[label], label=str(label))
        for label in unique_labels
    ]
    fig.legend(
        handles=legend_patches,
        title="Classes",
        loc="upper right",
        bbox_to_anchor=(1.15, 1)
    )
    
    plt.tight_layout()
    
    if saveimg:
        plt.savefig(saveimg, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {saveimg}")
    else:
        plt.show()