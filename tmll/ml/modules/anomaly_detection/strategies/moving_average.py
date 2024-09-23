import pandas as pd
import numpy as np
from typing import List, Tuple

from tmll.ml.modules.anomaly_detection.strategies.base import AnomalyDetectionStrategy

class MovingAverageStrategy(AnomalyDetectionStrategy):
    def detect_anomalies(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Detect anomalies using the moving average method.

        :param data: The data to analyze for anomalies
        :type data: pd.DataFrame
        :param kwargs: Additional parameters for the moving average method
        :type kwargs: dict
        :return: DataFrame with detected anomalies
        :rtype: pd.DataFrame
        """
        window_size = kwargs.get('moving_average_window_size', 10)
        threshold = kwargs.get('moving_average_threshold', 2)
        
        anomalies = pd.DataFrame(index=data.index)
        for column in data.columns:
            if column != 'timestamp':
                column_data = self._remove_minimum(data[[column]])
                moving_avg = column_data.rolling(window=window_size).mean()
                moving_std = column_data.rolling(window=window_size).std()
                anomalies[f'{column}_is_anomaly'] = np.abs(column_data - moving_avg) > (threshold * moving_std)

        anomaly_periods = self._identify_anomaly_periods(anomalies)
        return anomalies, anomaly_periods