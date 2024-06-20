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
        self.dataset = dataset.copy()
        self.method = method
        self.threshold = threshold

    def remove_outliers(self, target_features: list[str] = []) -> pd.DataFrame:
        """
        Remove the outliers based on the selected method.

        Returns:
            pd.DataFrame: The data without outliers.
        """
        if self.method == "zscore":
            z_scores = zscore(self.dataset)
            self.dataset = self.dataset[(
                z_scores < self.threshold).all(axis=1)]
        elif self.method == "iqr":
            q1 = self.dataset.quantile(0.25)
            q3 = self.dataset.quantile(0.75)
            iqr_values = iqr(self.dataset)
            self.dataset = self.dataset[~((self.dataset < (
                q1 - 1.5 * iqr_values)) | (self.dataset > (q3 + 1.5 * iqr_values))).any(axis=1)]
        else:
            raise ValueError(
                f"The remove outliers method is not among the available remove outliers methods, which are {', '.join(AVAILABLE_OUTLIERS_METHODS)}.")

        return self.dataset
