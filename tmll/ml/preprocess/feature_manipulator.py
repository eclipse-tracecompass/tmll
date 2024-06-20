"""
This module is for manipulating the features of the data. This module can be used to manipulate the features of the data before applying machine learning algorithms to the data.

The following feature manipulation methods may be used:
    - Basic Manipulation: For basic manipulation of the features of the data.
        - Remove Features: For removing the selected features from the data.
        - Select Features: For selecting the selected features from the data.
    - Feature Selection: For selecting the most important features from the data.
        - PCA: For selecting the most important features based on the Principal Component Analysis.
        - SelectKBest: For selecting the most important features based on the SelectKBest method.
        - TSNE: For selecting the most important features based on the t-Distributed Stochastic Neighbor Embedding.

Author: Kaveh Shahedi
"""

import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class FeatureManipulator:

    class Basic:

        def __init__(self, dataset: pd.DataFrame) -> None:
            self.dataset = dataset.copy()

        def remove_features(self, features: list[str]) -> pd.DataFrame:
            """
            Remove the selected features from the data.

            Returns:
                pd.DataFrame: The data without the selected features.
            """
            return self.dataset.drop(columns=features)

        def select_features(self, features: list[str]) -> pd.DataFrame:
            """
            Select the features from the data.

            Returns:
                pd.DataFrame: The data with the selected features.
            """
            return self.dataset[features]

    class Selection:

        AVAILABLE_SELECTION_METHODS = ["pca", "selectkbest", "tsne"]

        def __init__(self, dataset: pd.DataFrame, method: str = "pca", k: int = 5) -> None:
            self.dataset = dataset.copy()
            self.method = method
            self.k = k

        def select_features(self, target_feature: str) -> pd.DataFrame:
            """
            Select the most important features based on the selected method.

            Returns:
                pd.DataFrame: The data with the most important features.
            """
            if self.method == "pca":
                pca = PCA(n_components=self.k)
                self.dataset = pd.DataFrame(pca.fit_transform(self.dataset))
            elif self.method == "selectkbest":
                selector = SelectKBest(f_classif, k=self.k)
                self.dataset = pd.DataFrame(selector.fit_transform(
                    self.dataset, self.dataset[target_feature]))
            elif self.method == "tsne":
                tsne = TSNE(n_components=self.k)
                self.dataset = pd.DataFrame(tsne.fit_transform(self.dataset))
            else:
                raise ValueError(
                    f"The selection method is not among the available selection methods, which are {', '.join(self.AVAILABLE_SELECTION_METHODS)}.")

            return self.dataset
