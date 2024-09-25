import pandas as pd

from tmll.ml.preprocess.normalizer import Normalizer

class DataPreprocessor:
    @staticmethod
    def normalize(dataframe: pd.DataFrame, normalization_method: str = 'minmax') -> pd.DataFrame:
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
    
    @staticmethod
    def resample(dataframe: pd.DataFrame, frequency: str = '1h') -> pd.DataFrame:
        """
        Resample the DataFrame to the specified frequency.

        :param dataframe: The DataFrame to resample
        :type dataframe: pd.DataFrame
        :param frequency: The frequency to resample the timestamps to (e.g., '1h' for hourly or '1d' for daily), defaults to '1H'
        :type frequency: str, optional
        :return: The resampled DataFrame
        :rtype: pd.DataFrame
        """
        return dataframe.resample(frequency).mean().interpolate()
    
    @staticmethod
    def trim_dataframe(dataframe: pd.DataFrame, threshold: float = 0.01, min_active_period: int = 5):
        """
        Trim the dataframe to remove inactive periods at the start and end,
        while preserving any gaps in the middle.
        
        :param df: Input dataframe with timestamp index
        :type df: pd.DataFrame
        :param threshold: Minimum value to consider as active data
        :type threshold: float, optional, defaults to 0.01
        :param min_active_period: Minimum number of consecutive active rows to consider
        :type min_active_period: int, optional, defaults to 5
        :return: Trimmed dataframe
        """
        # Create a boolean mask for active data
        active_mask = (dataframe > threshold).any(axis=1)
    
        # Find the first and last indices of active data
        active_periods = active_mask.rolling(window=min_active_period).sum() >= min_active_period
        
        if active_periods.any():
            start_index = active_periods.idxmax()
            end_index = active_periods[::-1].idxmax()

            """
            Here we want to trim the dataframe to include only the active data in 
            a way that we only remove the beginning and end inactive periods, while
            preserving any gaps in the middle. We can do this by finding the first
            and last active data points and returning the dataframe between them.
            """
            
            # Find the first True value after start_index
            first_active = active_mask.loc[start_index:].idxmax()
            
            # Find the last True value before end_index
            last_active = active_mask.loc[:end_index][::-1].idxmax()
            
            return dataframe.loc[first_active:last_active]
        else:
            return dataframe.iloc[0:0]