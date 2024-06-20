"""
This module contains the Encoder class which is used to encode the data.

The following encoding methods may be used:
    - One-Hot: For encoding the categorical features based on the one-hot encoding.
    - Ordinal: For encoding the categorical features based on the ordinal encoding.

Author: Kaveh Shahedi
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

AVAILABLE_ENCODE_METHODS = ["onehot", "ordinal"]


class Encoder:

    def __init__(self, dataset: pd.DataFrame, method: str = "onehot") -> None:
        self.dataset = dataset.copy()
        self.method = method

    def encode(self, target_features: list[str] = []) -> pd.DataFrame:
        """
        Encode the data based on the selected method.

        Returns:
            pd.DataFrame: The encoded data.
        """
        if self.method == "onehot":
            encoder = OneHotEncoder(
                sparse=False, drop="first", handle_unknown="error")
        elif self.method == "ordinal":
            encoder = OrdinalEncoder()
        else:
            raise ValueError(
                f"The encode method is not among the available encode methods, which are {', '.join(AVAILABLE_ENCODE_METHODS)}.")

        # If target features are not provided, encode all the features, otherwise encode the target features
        if len(target_features) == 0:
            self.dataset = pd.DataFrame(encoder.fit_transform(
                self.dataset), columns=self.dataset.columns)
        else:
            self.dataset[target_features] = encoder.fit_transform(
                self.dataset[target_features])

        return self.dataset
