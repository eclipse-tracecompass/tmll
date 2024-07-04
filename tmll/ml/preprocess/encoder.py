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
        """Initialize the Encoder class.

        Args:
            dataset (pd.DataFrame): The dataset to be encoded.
            method (str, optional): The encoding method. Use get_encoding_methods() to see the available encoding methods. Defaults to "onehot".
        """
        self.dataset = dataset.copy()
        self.method = method

    def encode(self, target_features: list[str] = []) -> pd.DataFrame:
        """Encode the data based on the selected method (useful for encoding the categorical features)

        Args:
            target_features (list[str], optional): Which features to be encoded. If empty, all the dataset will be encoded. Defaults to [].

        Raises:
            ValueError: If the encode method is not among the available encode methods.

        Returns:
            pd.DataFrame: The encoded dataset.
        """

        if self.method == "onehot":
            encoder = OneHotEncoder(drop="first", handle_unknown="ignore")
        elif self.method == "ordinal":
            encoder = OrdinalEncoder()
        else:
            raise ValueError(f"The encode method is not among the available encode methods, which are {', '.join(AVAILABLE_ENCODE_METHODS)}.")

        # If target features are not provided, encode all the features, otherwise encode the target features
        if len(target_features) == 0:
            self.dataset = pd.DataFrame(encoder.fit_transform(self.dataset), columns=self.dataset.columns)
        else:
            for feature in target_features:
                self.dataset[feature] = encoder.fit_transform(self.dataset[[feature]]).toarray() # type: ignore

        return self.dataset

    @staticmethod
    def get_encoding_methods() -> list[str]:
        """Get the available encoding methods.

        Returns:
            list[str]: The available encoding methods.
        """
        return AVAILABLE_ENCODE_METHODS
