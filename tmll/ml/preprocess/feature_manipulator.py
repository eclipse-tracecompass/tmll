"""
This module is for manipulating the features of the data. This module can be used to manipulate the features of the data before applying machine learning algorithms to the data.

The following feature manipulation methods may be used:
    - Basic Manipulation: For basic manipulation of the features of the data.
        - Remove Features: For removing the selected features from the data.
        - Keep Features: For selecting and keeping the selected features from the data.
    - Feature Selection: For selecting the most important features from the data.
        - PCA: For selecting the most important features based on the Principal Component Analysis.
        - SelectKBest: For selecting the most important features based on the SelectKBest method.
        - TSNE: For selecting the most important features based on the t-Distributed Stochastic Neighbor Embedding.

Author: Kaveh Shahedi
"""

from typing import Optional
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class FeatureManipulator:

    class Basic:

        def __init__(self, dataset: pd.DataFrame) -> None:
            self.dataset = dataset.copy()

        def remove_features(self, features: list[str]) -> pd.DataFrame:
            """Removing the chosen features from the data.

            Args:
                features (list[str]): List of considered features to be removed.

            Returns:
                pd.DataFrame: Dataset without the removed features.
            """

            return self.dataset.drop(columns=features)

        def keep_features(self, features: list[str]) -> pd.DataFrame:
            """Selecting and keeping the chosen features from the data.

            Args:
                features (list[str]): List of considered features to be kept.

            Returns:
                pd.DataFrame: Dataset with the kept features.
            """

            return self.dataset[features]

    class Selection:

        AVAILABLE_SELECTION_METHODS = ["pca", "selectkbest", "tsne"]

        def __init__(self, dataset: pd.DataFrame, method: str = "pca", k: int = 5) -> None:
            self.dataset = dataset.copy()
            self.method = method
            self.k = k

        def select_features(self, target_feature: Optional[str] = None) -> pd.DataFrame:
            """Selecting the most important features from the data (statistically).

            Args:
                target_feature (str): The target feature to be considered for the feature selection. You may consider the target feature as the dependent variable.

            Raises:
                ValueError: If the selection method is not among the available selection methods.
                ValueError: If the target feature is not provided for the methods that require the target feature.

            Returns:
                pd.DataFrame: Dataset with the selected features.
            """

            if self.method == "pca":
                pca = PCA(n_components=self.k)
                self.dataset = pd.DataFrame(pca.fit_transform(self.dataset))
            elif self.method == "selectkbest":
                # Check if the target feature is provided
                if target_feature is None:
                    raise ValueError("The target feature should be provided for the SelectKBest method.")

                selector = SelectKBest(f_classif, k=self.k)
                self.dataset = pd.DataFrame(selector.fit_transform(self.dataset, self.dataset[target_feature]))
            elif self.method == "tsne":
                tsne = TSNE(n_components=self.k)
                self.dataset = pd.DataFrame(tsne.fit_transform(self.dataset))
            else:
                raise ValueError(f"The selection method is not among the available selection methods, which are {', '.join(self.AVAILABLE_SELECTION_METHODS)}.")

            return self.dataset
