from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from fuzzytree import FuzzyDecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import numpy as np


# Ignore warnings for DT https://stackoverflow.com/questions/29086398/sklearn-turning-off-warnings
import warnings 
warnings.filterwarnings("ignore")


def addlabels(x, y):
    """
    Add text labels on top of bars or points in a 1D plot.

    Parameters
    ----------
    x : list or array-like
        X-axis positions or labels (must be indexable).
    y : list or array-like
        Y values to annotate.

    Notes
    -----
    This function assumes you're already inside a plotting context
    (e.g., after plt.plot or plt.bar). It writes the numeric value of `y`
    above each x-position.
    """
    for i in range(len(x)):
        plt.text(i, round(y[i], 4), round(y[i], 4), ha='center')


class SimpleClassifiers:
    """
    Wrapper for training and evaluating a collection of simple classifiers.

    This class:
    - Instantiates several sklearn-compatible classifiers by default
    - Fits all of them on the same training data
    - Predicts on train and test sets
    - Computes train/test accuracies
    - Provides helper methods to get indices of TP/TN/FP/FN on the training set
    - Provides plotting utilities for accuracies and confusion matrices

    Parameters
    ----------
    classifiers : list of estimators, optional
        List of sklearn-style classifier objects (with `.fit` and `.predict`).
        If None, a default set of classifiers is used:
        - DecisionTreeClassifier
        - RandomForestClassifier
        - LGBMClassifier
        - LogisticRegression
        - GaussianNB

    Attributes
    ----------
    classifiers : list
        The classifier objects.
    names : list of str
        Class names (e.g. 'RandomForestClassifier').
    models : dict
        Mapping from classifier name to fitted model (after `.fit`).
    X_train, y_train : array-like
        Training data and labels (stored after `.fit`).
    X_test : array-like
        Test data (stored after `.predict`).
    y_train_p, y_test_p : dict
        Predicted labels on train/test for each model.
    train_accuracies, test_accuracies : dict
        Accuracy scores keyed by classifier name (after `.calculate_accuracies`).
    """

    def __init__(self, classifiers=None):
        if classifiers is None:
            classifiers = [
                DecisionTreeClassifier(),         # (optionally configure max_depth)
                # FuzzyDecisionTreeClassifier(),  # uncomment to include
                RandomForestClassifier(),         # (optionally configure n_estimators, max_depth)
                LGBMClassifier(verbose=-1),       # Ignore LightGBM warnings
                # XGBClassifier(),                # uncomment to include
                LogisticRegression(),             # may need solver='liblinear' for small / binary
                # MLPClassifier(),                # uncomment to include
                GaussianNB(),
            ]
        self.classifiers = classifiers
        self.names = [classifier.__class__.__name__ for classifier in self.classifiers]

    def fit(self, X_train, y_train):
        """
        Fit all classifiers on the training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : SimpleClassifiers
            Fitted object.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.models = {}
        for classifier in self.classifiers:
            self.models[classifier.__class__.__name__] = classifier.fit(self.X_train, self.y_train)
        return self

    def predict(self, X_test):
        """
        Predict labels for both training and test sets for all models.

        Parameters
        ----------
        X_test : array-like of shape (n_samples_test, n_features)
            Test feature matrix.

        Returns
        -------
        y_test_p : dict
            Dictionary mapping classifier name -> predicted labels on X_test.
        """
        self.X_test = X_test
        self.y_train_p = {}
        self.y_test_p = {}
        for name, model in self.models.items():
            self.y_train_p[name] = model.predict(self.X_train)
            self.y_test_p[name] = model.predict(self.X_test)
        return self.y_test_p
    
    def calculate_accuracies(self, y_test):
        """
        Compute train and test accuracy for all classifiers.

        Parameters
        ----------
        y_test : array-like of shape (n_samples_test,)
            True labels for X_test (must correspond to the last call to `.predict`).

        Returns
        -------
        test_accuracies : dict
            Mapping classifier name -> test accuracy.
        """
        self.y_test = y_test
        self.train_accuracies = {}
        self.test_accuracies = {}
        for name in self.names:
            self.train_accuracies[name] = accuracy_score(self.y_train, self.y_train_p[name])
            self.test_accuracies[name] = accuracy_score(self.y_test, self.y_test_p[name])
        return self.test_accuracies

    
    def compute_indices(self, index_type):
        """
        Generic method to compute TP/TN/FP/FN indices on the training set.

        Parameters
        ----------
        index_type : {'tp_tn', 'tp', 'tn', 'fp', 'fn'}
            Type of indices to compute:
            - 'tp_tn': all correctly classified (both TP and TN)
            - 'tp'   : true positives (y_true == y_pred == 1)
            - 'tn'   : true negatives (y_true == y_pred == 0)
            - 'fp'   : false positives (y_true == 0, y_pred == 1)
            - 'fn'   : false negatives (y_true == 1, y_pred == 0)

        Returns
        -------
        indices : dict
            Mapping classifier name -> list of indices.
        """
        valid_types = {"tp_tn", "fp", "tn", "fn", "tp"}
        if index_type not in valid_types:
            raise ValueError(f"Invalid index_type. Choose from {valid_types}.")
        
        indices = {}
        for name in self.names:
            if index_type == "tp_tn":
                indices[name] = [
                    index
                    for index, (y_true, y_p) in enumerate(zip(self.y_train, self.y_train_p[name]))
                    if y_true == y_p and (y_true == 1 or y_true == 0)
                ]
            elif index_type == "tp":
                indices[name] = [
                    index
                    for index, (y_true, y_p) in enumerate(zip(self.y_train, self.y_train_p[name]))
                    if y_true == y_p and y_true == 1
                ]
            elif index_type == "tn":
                indices[name] = [
                    index
                    for index, (y_true, y_p) in enumerate(zip(self.y_train, self.y_train_p[name]))
                    if y_true == y_p and y_true == 0
                ]
            elif index_type == "fp":
                indices[name] = [
                    index
                    for index, (y_true, y_p) in enumerate(zip(self.y_train, self.y_train_p[name]))
                    if y_true != y_p and y_true == 0
                ]
            elif index_type == "fn":
                indices[name] = [
                    index
                    for index, (y_true, y_p) in enumerate(zip(self.y_train, self.y_train_p[name]))
                    if y_true != y_p and y_true == 1
                ]
        return indices
    
    def plot_accuracies(self,
                        ax,
                        color=None,
                        linelabel='',
                        title=None):
        """
        Plot train and test accuracies for all classifiers.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object on which to draw the plot.
        color : str or None, optional
            Color used for the lines (train and test).
        linelabel : str, optional
            Suffix added to legend labels (e.g., experiment name).
        title : str or None, optional
            Title for the plot.

        Notes
        -----
        Requires `.calculate_accuracies()` to have been called beforehand.
        """
        y1 = list(self.train_accuracies.values())
        y2 = list(self.test_accuracies.values())
        
        plt.plot(
            self.names,
            y1,
            color=color,
            label="Train" if not linelabel else "Train - " + linelabel
        )
        addlabels(self.names, y1)
        plt.plot(
            self.names,
            y2,
            color=color,
            linestyle='--',
            label="Test" if not linelabel else "Test - " + linelabel
        )
        addlabels(self.names, y2)
        
        ax.set_ylim(0, 1)
        plt.xticks(rotation=90)

        ax.set_title(title)
        plt.legend()
        plt.tight_layout()

    def get_model(self, i):
        """
        Get the i-th fitted model.

        Parameters
        ----------
        i : int
            Index of the classifier in the internal list.

        Returns
        -------
        model : estimator
            The fitted model object.
        """
        return list(self.models.values())[i]

    def plot_confusion_matrices(self, y_pred_dict, y_test, norm_type='true'):
        """
        Plot confusion matrices (as heatmaps) using externally provided y_pred.

        Parameters
        ----------
        y_test : array-like of shape (n_samples,)
            True labels.
        y_pred_dict : dict
            Dictionary mapping classifier name -> predicted labels.
            Example: {"RandomForestClassifier": y_pred_rf, ...}
        norm_type : {'true', 'pred', 'all', None}, optional
            Normalization mode:
            - 'true': normalize by true label counts (rows sum to 1)
            - 'pred': normalize by predicted label counts (columns sum to 1)
            - 'all' : divide by total number of samples
            - anything else: no normalization

        Notes
        -----
        Creates a 2×3 grid of subplots; if more than 6 classifiers exist,
        only the first 6 are shown.
        """

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        axes = axes.flatten()

        for i, name in enumerate(self.names):
            if i >= len(axes):
                break

            if name not in y_pred_dict:
                raise ValueError(f"Missing predictions for classifier '{name}' in y_pred_dict.")

            y_pred = y_pred_dict[name]
            cm = normalize_confusion_matrix(y_test, y_pred, norm_type)

            sns.heatmap(cm, annot=True, fmt='.5f', ax=axes[i], cmap='Blues')
            axes[i].set_title(name)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')

        plt.tight_layout()
        plt.show()



def normalize_confusion_matrix(y_true, y_pred, norm_type):
    """
    Compute and normalize a confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    norm_type : {'true', 'pred', 'all', other}
        Normalization mode:
        - 'true': normalize each row (by true label count)
        - 'pred': normalize each column (by predicted label count)
        - 'all' : divide by total count
        - any other: no normalization

    Returns
    -------
    cm_normalized : ndarray of shape (n_classes, n_classes)
        Normalized confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    if norm_type == 'true':
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif norm_type == 'pred':
        cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    elif norm_type == 'all':
        cm_normalized = cm.astype('float') / cm.sum()
    else:
        cm_normalized = cm.astype('float')

    return cm_normalized



