from typing import Dict, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate

from lightgbm import LGBMClassifier
from fuzzytree import FuzzyDecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")  # e.g. DT, LGBM, etc.


def addlabels(x, y, ax: Optional[plt.Axes] = None):
    """
    Add numeric value labels on top of bars or points in a 1D plot.
    """
    if ax is None:
        ax = plt.gca()
    for i, val in enumerate(y):
        ax.text(i, round(val, 4), round(val, 4), ha="center")


class SimpleClassifiers:
    """
    Wrapper for training, evaluating, and cross-validating a collection of classifiers.

    Parameters
    ----------
    classifiers : dict or list, optional
        - If dict: {name: estimator}
        - If list: list of sklearn-style estimators; names will be class names.
        If None, a default set of classifiers is used.

    Attributes
    ----------
    models : dict
        {name: estimator} of fitted models after .fit().
    names : list of str
        Model names in evaluation order.
    train_accuracies, test_accuracies : dict
        Filled after .calculate_accuracies().
    cv_results_ : dict
        Filled after .cross_validate(). {name: cross_validate(...) output}
    """

    def __init__(self, classifiers: Optional[Union[Dict[str, object], List[object]]] = None):
        if classifiers is None:
            default_list = [
                DecisionTreeClassifier(),
                # FuzzyDecisionTreeClassifier(),
                RandomForestClassifier(),
                LGBMClassifier(verbose=-1),
                # XGBClassifier(),
                LogisticRegression(),
                # MLPClassifier(),
                GaussianNB(),
            ]
            self.models = {clf.__class__.__name__: clf for clf in default_list}
        elif isinstance(classifiers, dict):
            self.models = classifiers
        else:  # assume list
            self.models = {clf.__class__.__name__: clf for clf in classifiers}

        self.names = list(self.models.keys())

        # slots to be filled later
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_train_p = {}
        self.y_test_p = {}
        self.train_accuracies = {}
        self.test_accuracies = {}
        self.cv_results_ = {}

    # ---------------------------------------------------------------------
    # Fit / Predict / Accuracy
    # ---------------------------------------------------------------------
    def fit(self, X_train, y_train):
        """Fit all classifiers on the training data."""
        self.X_train = X_train
        self.y_train = y_train

        for name, clf in self.models.items():
            self.models[name] = clf.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        """
        Predict labels for both training and test sets for all models.

        Returns
        -------
        y_test_p : dict
            {model_name: y_pred_on_X_test}
        """
        self.X_test = X_test

        # train predictions (for indices / train accuracy)
        self.y_train_p = {
            name: model.predict(self.X_train) for name, model in self.models.items()
        }

        # test predictions
        self.y_test_p = {
            name: model.predict(self.X_test) for name, model in self.models.items()
        }

        return self.y_test_p

    def calculate_accuracies(self, y_test):
        """
        Compute train and test accuracy for all classifiers.
        Assumes .predict() has been called.
        """
        self.y_test = y_test

        self.train_accuracies = {
            name: accuracy_score(self.y_train, self.y_train_p[name])
            for name in self.names
        }
        self.test_accuracies = {
            name: accuracy_score(self.y_test, self.y_test_p[name])
            for name in self.names
        }

        return self.test_accuracies

    # ---------------------------------------------------------------------
    # Cross-validation
    # ---------------------------------------------------------------------
    def cross_validate(
        self,
        X,
        y,
        cv=5,
        scoring: Union[str, List[str], Dict[str, str]] = "accuracy, precision, recall, f1, roc_auc",
        n_jobs: Optional[int] = None,
        return_estimator: bool = False,
    ):
        """
        Run sklearn-style cross-validation for each classifier.

        Parameters
        ----------
        X, y : array-like
            Full dataset.
        cv : int, CV splitter, or iterable, default=5
            As in sklearn.model_selection.cross_validate.
        scoring : str, list, or dict, default='accuracy'
            Scoring metric(s).
        n_jobs : int or None, optional
            Number of parallel jobs.
        return_estimator : bool, default=False
            Whether to return fitted estimators for each split.

        Returns
        -------
        cv_results_ : dict
            {model_name: cross_validate(...) output}
        """
        self.cv_results_ = {}
        for name, clf in self.models.items():
            self.cv_results_[name] = cross_validate(
                clone(clf),
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                return_estimator=return_estimator,
            )
        return self.cv_results_

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def compute_indices(self, index_type: str):
        """
        Get indices of TP/TN/FP/FN on the training set.

        index_type : {'tp_tn', 'tp', 'tn', 'fp', 'fn'}
        """
        valid_types = {"tp_tn", "fp", "tn", "fn", "tp"}
        if index_type not in valid_types:
            raise ValueError(f"Invalid index_type. Choose from {valid_types}.")

        indices = {}
        y_true = np.asarray(self.y_train)

        for name in self.names:
            y_pred = np.asarray(self.y_train_p[name])

            if index_type == "tp_tn":
                mask = (y_true == y_pred) & np.isin(y_true, [0, 1])
            elif index_type == "tp":
                mask = (y_true == y_pred) & (y_true == 1)
            elif index_type == "tn":
                mask = (y_true == y_pred) & (y_true == 0)
            elif index_type == "fp":
                mask = (y_true != y_pred) & (y_true == 0)
            else:  # 'fn'
                mask = (y_true != y_pred) & (y_true == 1)

            indices[name] = np.where(mask)[0].tolist()

        return indices

    def plot_accuracies(
        self,
        ax: plt.Axes,
        color: Optional[str] = None,
        linelabel: str = "",
        title: Optional[str] = None,
    ):
        """
        Plot train and test accuracies for all classifiers on a given Axes.
        Assumes .calculate_accuracies() was called.
        """
        y_train_vals = [self.train_accuracies[name] for name in self.names]
        y_test_vals = [self.test_accuracies[name] for name in self.names]

        label_train = "Train" if not linelabel else f"Train - {linelabel}"
        label_test = "Test" if not linelabel else f"Test - {linelabel}"

        ax.plot(self.names, y_train_vals, label=label_train, color=color)
        addlabels(self.names, y_train_vals, ax=ax)

        ax.plot(self.names, y_test_vals, linestyle="--", label=label_test, color=color)
        addlabels(self.names, y_test_vals, ax=ax)

        ax.set_ylim(0, 1)
        ax.set_xticklabels(self.names, rotation=90)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

    def get_model(self, i: int):
        """Return the i-th fitted model."""
        return self.models[self.names[i]]

    def plot_confusion_matrices(self, y_pred_dict, y_test, norm_type: str = "true"):
        """
        Plot confusion matrices (heatmaps) using externally provided predictions.

        y_pred_dict : dict
            {model_name: y_pred}
        norm_type : {'true', 'pred', 'all', None}
        """
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        axes = axes.flatten()

        for i, name in enumerate(self.names):
            if i >= len(axes):
                break

            if name not in y_pred_dict:
                raise ValueError(
                    f"Missing predictions for classifier '{name}' in y_pred_dict."
                )

            y_pred = y_pred_dict[name]
            cm = normalize_confusion_matrix(y_test, y_pred, norm_type)

            sns.heatmap(cm, annot=True, fmt=".5f", ax=axes[i], cmap="Blues")
            axes[i].set_title(name)
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")

        plt.tight_layout()
        plt.show()


def normalize_confusion_matrix(y_true, y_pred, norm_type: str):
    """
    Compute and normalize a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    cm = cm.astype("float")
    if norm_type == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm, row_sums, where=row_sums != 0)
    elif norm_type == "pred":
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_normalized = np.divide(cm, col_sums, where=col_sums != 0)
    elif norm_type == "all":
        total = cm.sum()
        cm_normalized = cm / total if total != 0 else cm
    else:
        cm_normalized = cm

    return cm_normalized
