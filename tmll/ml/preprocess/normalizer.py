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
        """Initialize the Normalizer class.

        Args:
            dataset (pd.DataFrame): The dataset to be normalized.
            method (str, optional): The normalization method. Use get_normalize_methods() to see the available normalization methods. Defaults to "standard".
        """

        self.dataset = dataset.copy()
        self.method = method

    def normalize(self, target_features: list[str] = []) -> pd.DataFrame:
        """Normalize the data based on the selected method.

        Args:
            target_features (list[str], optional): Which features to be normalized. If empty, all the dataset will be normalized. Defaults to [].

        Raises:
            ValueError: If the normalize method is not among the available normalize methods.

        Returns:
            pd.DataFrame: The normalized dataset.
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

    @staticmethod
    def get_normalize_methods() -> list[str]:
        """Get the available normalization methods.

        Returns:
            list[str]: The available normalization methods.
        """
        return AVAILABLE_NORMALIZE_METHODS
