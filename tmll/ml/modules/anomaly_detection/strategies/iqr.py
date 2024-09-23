import pandas as pd
from typing import List, Tuple

from tmll.ml.modules.anomaly_detection.strategies.base import AnomalyDetectionStrategy

class IQRStrategy(AnomalyDetectionStrategy):
    def detect_anomalies(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Detect anomalies using the interquartile range (IQR) method.

        :param data: The data to analyze for anomalies
        :type data: pd.DataFrame
        :param kwargs: Additional parameters for the IQR method
        :type kwargs: dict
        :return: DataFrame with detected anomalies
        :rtype: pd.DataFrame
        """
        anomalies = pd.DataFrame(index=data.index)
        for column in data.columns:
            if column != 'timestamp':
                column_data = self._remove_minimum(data[[column]])
                Q1 = column_data.quantile(0.25)
                Q3 = column_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                anomalies[f'{column}_is_anomaly'] = column_data > upper_bound

        anomaly_periods = self._identify_anomaly_periods(anomalies)
        return anomalies, anomaly_periods
