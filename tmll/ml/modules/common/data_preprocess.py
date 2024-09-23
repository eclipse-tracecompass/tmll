import pandas as pd

from tmll.ml.preprocess.normalizer import Normalizer

class DataPreprocessor:
    @staticmethod
    def preprocess(dataframe: pd.DataFrame, normalization_method: str = 'minmax') -> pd.DataFrame:
        """
        Preprocess the data by normalizing it.

        :param dataframe: The DataFrame to preprocess
        :type dataframe: pd.DataFrame
        :param normalization_method: The normalization method to use, defaults to 'minmax'
        :type normalization_method: str, optional
        :return: The preprocessed DataFrame
        :rtype: pd.DataFrame
        """
        normalizer = Normalizer(dataset=dataframe, method=normalization_method)
        return normalizer.normalize(target_features=[col for col in dataframe.columns if col != 'timestamp'])