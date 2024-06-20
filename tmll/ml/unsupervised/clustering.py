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

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import cut_tree

from tmll.ml.preprocess.normalizer import Normalizer
from tmll.ml.preprocess.outlier_remover import OutlierRemover
from tmll.ml.preprocess.encoder import Encoder
from tmll.ml.preprocess.feature_manipulator import FeatureManipulator

AVALIABLE_MODELS = ["kmeans", "dbscan", "hierarchical", "spectral"]


class Clustering:

    def __init__(self, dataset: pd.DataFrame, model: str = "kmeans",
                 n_clusters: int = 3, optimal_n_clusters: bool = False,
                 categorical_features: list[str] = [], encoding_method: str = "onehot",
                 ignore_features: list[str] = [], keep_features: list[str] = [],
                 normalize: bool = False, normalize_method: str = "standard",
                 remove_outliers: bool = False, remove_outliers_method: str = "zscore",
                 random_state: int = 42):

        self.dataset = dataset
        self.model = model
        self.n_clusters = n_clusters
        self.optimal_n_clusters = optimal_n_clusters
        self.categorical_features = categorical_features
        self.ignore_features = ignore_features
        self.keep_features = keep_features
        self.encoding_method = encoding_method
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.remove_outliers = remove_outliers
        self.remove_outliers_method = remove_outliers_method
        self.random_state = random_state

        # If number of clusters is less than 2, raise an error
        if self.n_clusters < 2:
            raise ValueError(
                "The number of clusters should be greater than or equal to 2.")

        # If the number of features is less than the number of clusters, raise an error
        if len(self.dataset.columns) < self.n_clusters:
            raise ValueError(
                "The number of features is less than the number of clusters.")

        # If the model is not among the available models, raise an error
        if self.model not in AVALIABLE_MODELS:
            raise ValueError(
                f"The model is not among the available models, which are {', '.join(AVALIABLE_MODELS)}.")

        if len(self.ignore_features) > 0:
            self.dataset = FeatureManipulator.Basic(
                self.dataset).remove_features(self.ignore_features)

        if len(self.keep_features) > 0:
            self.dataset = FeatureManipulator.Basic(
                self.dataset).select_features(self.keep_features)

        if len(self.categorical_features) > 0:
            self.dataset = Encoder(
                self.dataset, method=self.encoding_method).encode(self.categorical_features)

        if normalize:
            self.dataset = Normalizer(
                self.dataset, method=self.normalize_method).normalize()

        if remove_outliers:
            self.dataset = OutlierRemover(
                self.dataset, method=self.remove_outliers_method).remove_outliers()
