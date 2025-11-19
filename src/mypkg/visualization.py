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





import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE


def plot_tsne(
    X,
    y,
    perplexities=[5, 30, 50, 70, 100],
    n_components=2,
    saveimg=None,
):
    """
    Plot t-SNE visualizations with different perplexities and a legend for class labels.
    Layout uses up to 2 columns; additional plots go to new rows.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Class labels (integer or string).
    perplexities : list of float, optional
        List of perplexity values to use for t-SNE.
    n_components : int, optional (default=2)
        Number of t-SNE output dimensions.
        - 2 → 2D scatter plot
        - 3 → 3D scatter plot
        - >3 → TSNE runs but only prints the shape (no plot).
    saveimg : str or None, optional
        If not None, the figure is saved to this path (relative or absolute).
    """

    num_plots = len(perplexities)
    ncols = 2
    nrows = math.ceil(num_plots / ncols)

    # 3D support
    is_3d = (n_components == 3)

    # Create figure
    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))

    # Axes grid
    if n_components == 2:
        axes = [
            fig.add_subplot(nrows, ncols, i + 1)
            for i in range(num_plots)
        ]
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        axes = [
            fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            for i in range(num_plots)
        ]
    else:
        axes = None  # we won't plot, just print shapes later

    # Colors per class
    y = np.asarray(y)
    unique_labels = np.unique(y)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: color for label, color in zip(unique_labels, colors)}

    # t-SNE per perplexity
    for i, perplexity in enumerate(perplexities):
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
        )
        X_tsne = tsne.fit_transform(X)

        if n_components == 2:
            ax = axes[i]
            ax.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                c=[label_to_color[lbl] for lbl in y],
                s=6,
            )
            ax.set_title(f"t-SNE (perplexity={perplexity})")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        elif n_components == 3:
            ax = axes[i]
            ax.scatter(
                X_tsne[:, 0],
                X_tsne[:, 1],
                X_tsne[:, 2],
                c=[label_to_color[lbl] for lbl in y],
                s=8,
            )
            ax.set_title(f"t-SNE 3D (perplexity={perplexity})")
            ax.set_xlabel("Comp 1")
            ax.set_ylabel("Comp 2")
            ax.set_zlabel("Comp 3")

        else:
            print(f"t-SNE result for perplexity={perplexity} has shape: {X_tsne.shape}")
            print("Plotting skipped because n_components > 3.")

    # Legend
    if n_components in [2, 3]:
        legend_patches = [
            mpatches.Patch(color=label_to_color[label], label=str(label))
            for label in unique_labels
        ]
        fig.legend(
            handles=legend_patches,
            title="Classes",
            loc="upper right",
            bbox_to_anchor=(1.15, 1),
        )

    plt.tight_layout()

    # Save OR show
    if saveimg:
        # Create directory if needed
        outdir = os.path.dirname(saveimg)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        abs_path = os.path.abspath(saveimg)
        fig.savefig(abs_path, dpi=300, bbox_inches="tight")
        print(f"[plot_tsne] Figure saved to: {abs_path}")
        plt.close(fig)  # close so it doesn't also pop a window in some environments
    else:
        plt.show()
