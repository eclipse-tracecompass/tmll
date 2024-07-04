"""
A module for clustering algorithms. This module can be used to cluster the data into different groups.
The data may be fetched through the TSP client, and the clustering algorithms can be applied to the data.

The following clustering algorithms may be used:
    - KMeans: For clustering the data into K groups based on the mean of the data.
    - DBSCAN: For clustering the data into groups based on density where there is noise in the data.
    - Hierarchical: For clustering the data into groups based on a hierarchy (e.g., dendrogram).
    - Spectral: For clustering the data into groups based on the eigenvectors of the similarity matrix.

The following metrics may be used to evaluate the clustering algorithms:
    - Silhouette Score: For evaluating the clustering algorithms based on the mean intra-cluster distance and the mean nearest-cluster distance.
    - Davies-Bouldin Index: For evaluating the clustering algorithms based on the average similarity between each cluster and its most similar cluster.
    - Calinski-Harabasz Index: For evaluating the clustering algorithms based on the ratio of the sum of between-cluster dispersion to within-cluster dispersion.

The following methods may be used to visualize the clustering algorithms:
    - Scatter Plot: For visualizing the clustering algorithms based on the data points.
    - Dendrogram: For visualizing the clustering algorithms based on the hierarchy of the clusters.
    - Heatmap: For visualizing the clustering algorithms based on the similarity matrix.

Author: Kaveh Shahedi
"""

from typing import Dict
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import cut_tree

from tqdm import tqdm

from tmll.ml.preprocess.normalizer import Normalizer
from tmll.ml.preprocess.outlier_remover import OutlierRemover
from tmll.ml.preprocess.encoder import Encoder
from tmll.ml.preprocess.feature_manipulator import FeatureManipulator

from tmll.common.services.logger import Logger

AVALIABLE_MODELS = ["kmeans", "dbscan", "hierarchical", "spectral"]


class Clustering:

    def __init__(self, dataset: pd.DataFrame, model: str = "kmeans",
                 n_clusters: int = 3, optimal_n_clusters: bool = False,
                 categorical_features: list[str] = [],
                 auto_encoding: bool = True, encoding_method: str = "onehot",
                 ignore_features: list[str] = [], keep_features: list[str] = [],
                 normalize: bool = False, normalize_method: str = "standard",
                 remove_outliers: bool = False, remove_outliers_method: str = "zscore",
                 random_state: int = 42, verbose=True) -> None:

        self.dataset = dataset
        self.model = model
        self.n_clusters = n_clusters
        self.optimal_n_clusters = optimal_n_clusters
        self.categorical_features = categorical_features
        self.ignore_features = ignore_features
        self.keep_features = keep_features
        self.auto_encoding = auto_encoding
        self.encoding_method = encoding_method
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.remove_outliers = remove_outliers
        self.remove_outliers_method = remove_outliers_method
        self.random_state = random_state

        self.logger = Logger(name="Clustering", verbose=verbose)

        # If number of clusters is less than 2, raise an error
        if self.n_clusters < 2:
            raise ValueError("The number of clusters should be greater than or equal to 2.")

        # If the number of features is less than the number of clusters, raise an error
        if len(self.dataset) < self.n_clusters:
            raise ValueError("The number of features is less than the number of clusters.")

        # If the model is not among the available models, raise an error
        if self.model not in AVALIABLE_MODELS:
            raise ValueError(f"The model is not among the available models, which are {', '.join(AVALIABLE_MODELS)}.")

        if len(self.ignore_features) > 0:
            self.dataset = FeatureManipulator.Basic(self.dataset).remove_features(self.ignore_features)
            self.logger.info(f"Removed the following features: {', '.join(self.ignore_features)}")

        if len(self.keep_features) > 0:
            self.dataset = FeatureManipulator.Basic(self.dataset).keep_features(self.keep_features)
            self.logger.info(f"Kept the following features: {', '.join(self.keep_features)}")

        if len(self.categorical_features) > 0 or self.auto_encoding:
            target_columns = self.categorical_features if len(self.categorical_features) > 0 else self.dataset.select_dtypes(include=["object"]).columns.tolist()
            if len(target_columns) > 0:
                self.dataset = Encoder(self.dataset, method=self.encoding_method).encode(target_columns)
                self.logger.info(f"Encoded the following categorical features: {', '.join(target_columns)}")

        if normalize:
            self.dataset = Normalizer(self.dataset, method=self.normalize_method).normalize()
            self.logger.info(f"Normalized the data using the {self.normalize_method} method.")

        if remove_outliers:
            self.dataset = OutlierRemover(self.dataset, method=self.remove_outliers_method).remove_outliers()
            self.logger.info(f"Removed the outliers using the {self.remove_outliers_method} method.")

        self.logger.info(f"Initiated the clustering algorithm. Model: {self.model}, Number of clusters: {self.n_clusters}")

    def execute(self) -> Dict:
        # Apply the clustering algorithm
        self.cluster()

        # Evaluate the clustering algorithm
        silhouette, davies_bouldin, calinski_harabasz = self.evaluate()

        # Get the clusters of the data
        clusters = self.get_clusters()

        # Return the results
        return {
            "model": self.model,
            "n_clusters": self.n_clusters,
            "clusters": clusters,
            "evaluation": {
                "silhouette": silhouette,
                "davies_bouldin": davies_bouldin,
                "calinski_harabasz": calinski_harabasz
            }
        }

    def __get_optimal_n_clusters(self, max_n_clusters: int = 5) -> int:
        """Get the optimal number of clusters based on the silhouette score. This is useful when the we do not know how many clusters are in the data.

        Args:
            max_n_clusters (int, optional): The maximum number of clusters that the search algorithm should check. Defaults to 5.

        Returns:
            int: The optimal number of clusters.
        """

        silhouette_scores = []

        for i in tqdm(range(2, max_n_clusters + 1)):
            if i >= len(self.dataset):
                print("Stopping the search for the optimal number of clusters because the number of clusters is greater than or equal to the number of data points.")
                break

            model = KMeans(n_clusters=i, random_state=self.random_state)
            model.fit(self.dataset)
            silhouette_scores.append(silhouette_score(self.dataset, model.labels_))

        return silhouette_scores.index(max(silhouette_scores)) + 2

    def cluster(self):
        """Apply the clustering algorithm based on the model.
        """

        if self.optimal_n_clusters:
            self.n_clusters = self.__get_optimal_n_clusters()
            self.logger.info(f"Optimal number of clusters: {self.n_clusters}")

        if self.model == "kmeans":
            self.model_object = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        elif self.model == "dbscan":
            self.model_object = DBSCAN(eps=0.5, min_samples=5)
        elif self.model == "hierarchical":
            self.model_object = AgglomerativeClustering(n_clusters=self.n_clusters)
        elif self.model == "spectral":
            self.model_object = SpectralClustering(n_clusters=self.n_clusters, random_state=self.random_state)
        else:
            raise ValueError(f"The model is not among the available models, which are {', '.join(AVALIABLE_MODELS)}.")

        self.logger.info(f"Applying the {self.model} clustering algorithm.")

        self.model_object.fit(self.dataset)

    def evaluate(self):
        """Evaluate the clustering algorithm based on the silhouette score, Davies-Bouldin index, and Calinski-Harabasz index.
        """

        if self.model_object is None:
            raise ValueError("The model is not defined. Please define the model before evaluating it.")

        silhouette = silhouette_score(self.dataset, self.model_object.labels_)
        davies_bouldin = davies_bouldin_score(self.dataset, self.model_object.labels_)
        calinski_harabasz = calinski_harabasz_score(self.dataset, self.model_object.labels_)

        self.logger.info(
            f"Evaluated the clustering algorithm. Silhouette Score: {silhouette}, Davies-Bouldin Index: {davies_bouldin}, Calinski-Harabasz Index: {calinski_harabasz}")

        return silhouette, davies_bouldin, calinski_harabasz

    def get_clusters(self) -> pd.DataFrame:
        """Get the clusters of the data.

        Returns:
            pd.DataFrame: The clusters of the data.
        """

        # Add the clusters to the dataset
        clusters = self.dataset.copy()
        clusters["cluster"] = self.model_object.labels_

        return clusters
