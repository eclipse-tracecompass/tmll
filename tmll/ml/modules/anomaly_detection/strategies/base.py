import pandas as pd
from typing import List, Tuple
from abc import ABC, abstractmethod

from tmll.common.services.logger import Logger

class AnomalyDetectionStrategy(ABC):
    """
    Abstract class for anomaly detection strategies.
    """

    def __init__(self):
        self.logger = Logger(self.__class__.__name__)

    @abstractmethod
    def detect_anomalies(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Detect anomalies in the given data.

        :param data: The data to analyze for anomalies
        :type data: pd.DataFrame
        :param kwargs: Additional parameters for the detection method
        :type kwargs: dict
        :return: Tuple of (DataFrame with detected anomalies, List of anomaly periods)
        :rtype: Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]
        """
        pass

    @staticmethod
    def _remove_minimum(data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the minimum value from each column of the data.

        :param data: The data to process
        :type data: pd.DataFrame
        :return: The data with minimum values removed from each column
        :rtype: pd.DataFrame
        """
        return data[data > data.min()]
    
    @staticmethod
    def _calculate_adaptive_window_size(data: pd.DataFrame) -> int:
        """
        Calculate an adaptive window size based on the input dataframe.

        :param data: The input dataframe
        :type data: pd.DataFrame
        :return: Adaptive window size
        :rtype: int
        """
        return max(int(len(data) * 0.005), 10)  # 0.5% of the data length or 10, whichever is larger

    @staticmethod
    def _identify_anomaly_periods(anomalies: pd.DataFrame, threshold: float = 0.9) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Identify continuous periods of anomalies.

        :param anomalies: DataFrame with boolean values indicating anomalies
        :type anomalies: pd.DataFrame
        :param threshold: The threshold for considering a period as anomalous
        :type threshold: float
        :return: List of tuples containing start and end timestamps of anomaly periods
        :rtype: List[Tuple[pd.Timestamp, pd.Timestamp]]
        """
        window_size = AnomalyDetectionStrategy._calculate_adaptive_window_size(anomalies)
        anomaly_periods = []
        start = None

        for i in range(len(anomalies) - window_size + 1):
            window = anomalies.iloc[i:i+window_size]
            if window.any(axis=1).mean() >= threshold and start is None:
                start = anomalies.index[i]
            elif window.any(axis=1).mean() < threshold and start is not None:
                anomaly_periods.append((start, anomalies.index[i+window_size-1]))
                start = None

        if start is not None:
            anomaly_periods.append((start, anomalies.index[-1]))

        return anomaly_periods