"""
This module is for normalizing the data. This module can be used to normalize the data before applying machine learning algorithms to the data.

The following normalization methods may be used:
    - Standard: For normalizing the data based on the mean and standard deviation of the data.
    - Min-Max: For normalizing the data based on the minimum and maximum values of the data.
    - Max-Abs: For normalizing the data based on the maximum absolute value of the data.
    - Robust: For normalizing the data based on the median and interquartile range of the data.

Author: Kaveh Shahedi    
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

AVAILABLE_NORMALIZE_METHODS = ["standard", "minmax", "maxabs", "robust"]


class Normalizer:

    def __init__(self, dataset: pd.DataFrame, method: str = "standard"):
        self.dataset = dataset.copy()
        self.method = method

    def normalize(self, target_features: list[str] = []) -> pd.DataFrame:
        """
        Normalize the data based on the selected method.

        Returns:
            pd.DataFrame: The normalized data.
        """
        if self.method == "standard":
            scaler = StandardScaler()
        elif self.method == "minmax":
            scaler = MinMaxScaler()
        elif self.method == "maxabs":
            scaler = MaxAbsScaler()
        elif self.method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(
                f"The normalize method is not among the available normalize methods, which are {', '.join(AVAILABLE_NORMALIZE_METHODS)}.")

        # If target features are not provided, normalize all the features, otherwise normalize the target features
        if len(target_features) == 0:
            self.dataset = pd.DataFrame(scaler.fit_transform(
                self.dataset), columns=self.dataset.columns)
        else:
            self.dataset[target_features] = scaler.fit_transform(
                self.dataset[target_features])

        return self.dataset
