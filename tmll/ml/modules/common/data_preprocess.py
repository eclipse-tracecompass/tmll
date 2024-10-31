from typing import Dict, Optional, Tuple
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

        # Normalize the columns that are neither 'timestamp' nor their type is not 'float64' or 'int64'
        target_features = []
        for column in dataframe.columns:
            if column != 'timestamp' and dataframe[column].dtype in ['float64', 'int64']:
                target_features.append(column)

        if not target_features:
            return dataframe
        
        return normalizer.normalize(target_features)
    
    @staticmethod
    def convert_to_datetime(dataframe: pd.DataFrame, timestamp_column: str = 'timestamp', set_index: bool = True) -> pd.DataFrame:
        """
        Convert the timestamp column to a datetime format.

        :param dataframe: The DataFrame to convert
        :type dataframe: pd.DataFrame
        :param timestamp_column: The name of the timestamp column, defaults to 'timestamp'
        :type timestamp_column: str, optional
        :param set_index: Whether to set the timestamp column as the index, defaults to True
        :type set_index: bool, optional
        :return: The DataFrame with the timestamp column converted to datetime
        :rtype: pd.DataFrame
        """
        dataframe[timestamp_column] = pd.to_datetime(dataframe[timestamp_column])
        if set_index:
            dataframe.set_index(timestamp_column, inplace=True)
        return dataframe
        
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
        
    @staticmethod
    def separate_timegraph(dataframe: pd.DataFrame, column: str) -> Dict[str, pd.DataFrame]:
        """
        Separate the DataFrame into multiple DataFrames based on the unique values in the specified column.

        :param dataframe: The DataFrame to separate
        :type dataframe: pd.DataFrame
        :param column: The column to separate the DataFrame by
        :type column: str
        :return: A dictionary containing the separated DataFrames
        :rtype: Dict[str, pd.DataFrame]
        """
        # If the column does not exist in the DataFrame, return the original DataFrame
        if column not in dataframe.columns:
            return {column: dataframe}
        
        # Separate the DataFrame based on the unique values in the specified column
        unique_values = dataframe[column].unique()
        return {value: dataframe[dataframe[column] == value].drop(columns=[column]) for value in unique_values}

    @staticmethod
    def align_timestamps(dataframes: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DatetimeIndex]]:
        """
        Align all dataframes to the reference index (from longest dataframe) 
        and fill missing values with zeros.

        :param dataframes: A dictionary of dataframes to align
        :type dataframes: Dict[str, pd.DataFrame]
        """
        # Get reference time index from the longest dataframe
        max_length = 0
        reference_index: Optional[pd.DatetimeIndex] = None
        
        for df in dataframes.values():
            if len(df.index) > max_length:
                if isinstance(df.index, pd.DatetimeIndex):
                    max_length = len(df.index)
                    reference_index = df.index
        
        # Reindex all dataframes to match reference and fill with zeros
        for name in dataframes:
            dataframes[name] = dataframes[name].reindex(reference_index, fill_value=0)
        
        return dataframes, reference_index