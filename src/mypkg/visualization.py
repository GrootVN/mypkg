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
    shap_values : array-like
        SHAP values for a multi-class classifier. Expected shape is
        (n_samples, n_features, n_classes) or equivalent after np.array().
    X : pandas.DataFrame
        Feature matrix used to compute SHAP values (columns = feature names).

    Notes
    -----
    - First, computes a mean absolute SHAP value per feature across all classes
      and samples, and prints the top 10 features.
    - Then generates SHAP summary plots for each class.
    """
    print("X_train1 shape:", X.shape)
    print("shap_values shape:", np.array(shap_values).shape)
     
    # Aggregate SHAP values across classes by taking the mean of absolute values across classes
    mean_abs_shap = np.mean(np.abs(shap_values).mean(axis=2), axis=0)

    # Create a DataFrame to pair feature names with their importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_abs_shap
    })

    # Sort features by importance
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)

    # Get the top 10 most significant features
    top_5_features = feature_importance.head(10)
    print("Top 10 most significant features:")
    print(top_5_features)
    
    # Plot SHAP summaries for all classes
    for class_index in range(shap_values.shape[2]):
        print(f"Summary plot for class {class_index}")
        shap.summary_plot(shap_values[..., class_index], features=X)


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