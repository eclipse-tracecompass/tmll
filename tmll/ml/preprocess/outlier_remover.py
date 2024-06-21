"""
This module is for removing outliers from the data. This module can be used to remove the outliers from the data before applying machine learning algorithms to the data.

The following outlier removal methods may be used:
    - Z-Score: For removing the outliers based on the Z-Score of the data.
    - IQR: For removing the outliers based on the Interquartile Range of the data.

Author: Kaveh Shahedi
"""

import pandas as pd

from scipy.stats import zscore, iqr

AVAILABLE_OUTLIERS_METHODS = ["zscore", "iqr"]


class OutlierRemover:

    def __init__(self, dataset: pd.DataFrame, method: str = "zscore", threshold: float = 3):
        """Initialize the OutlierRemover class.

        Args:
            dataset (pd.DataFrame): The dataset to remove the outliers from.
            method (str, optional): The outlier removing method. Use get_outlier_removal_methods() to see the available methods. Defaults to "zscore".
            threshold (float, optional): The threshold for removing the outliers (if needed). Defaults to 3.
        """

        self.dataset = dataset.copy()
        self.method = method
        self.threshold = threshold

    def remove_outliers(self, target_features: list[str] = []) -> pd.DataFrame:
        """Remove the outliers from the data based on the selected method.

        Args:
            target_features (list[str], optional): Which features to remove the outliers from. If empty, all the dataset will be used. Defaults to [].

        Raises:
            ValueError: If the remove outliers method is not among the available remove outliers methods.

        Returns:
            pd.DataFrame: The dataset without the outliers.
        """

        if self.method == "zscore":
            if len(target_features) == 0:
                self.dataset = self.dataset[(zscore(self.dataset) < self.threshold).all(axis=1)]
            else:
                self.dataset = self.dataset[(zscore(self.dataset[target_features]) < self.threshold).all(axis=1)]
        elif self.method == "iqr":
            if len(target_features) == 0:
                Q1 = self.dataset.quantile(0.25)
                Q3 = self.dataset.quantile(0.75)
                IQR = Q3 - Q1
                self.dataset = self.dataset[~((self.dataset < (Q1 - 1.5 * IQR)) | (self.dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
            else:
                Q1 = self.dataset[target_features].quantile(0.25)
                Q3 = self.dataset[target_features].quantile(0.75)
                IQR = Q3 - Q1
                self.dataset = self.dataset[~((self.dataset[target_features] < (Q1 - 1.5 * IQR)) | (self.dataset[target_features] > (Q3 + 1.5 * IQR))).any(axis=1)]
        else:
            raise ValueError(f"The remove outliers method is not among the available remove outliers methods, which are {', '.join(AVAILABLE_OUTLIERS_METHODS)}.")

        return self.dataset

    @staticmethod
    def get_outlier_removal_methods() -> list[str]:
        """Get the available outlier removal methods.

        Returns:
            list[str]: The available outlier removal methods.
        """
        return AVAILABLE_OUTLIERS_METHODS
