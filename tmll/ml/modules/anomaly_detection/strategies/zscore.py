import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy import stats

from tmll.ml.modules.anomaly_detection.strategies.base import AnomalyDetectionStrategy

class ZScoreStrategy(AnomalyDetectionStrategy):
    def detect_anomalies(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Detect anomalies using the Z-score method.

        :param data: The data to analyze for anomalies
        :type data: pd.DataFrame
        :param kwargs: Additional parameters for the Z-score method
        :type kwargs: dict
        :return: DataFrame with detected anomalies
        :rtype: pd.DataFrame
        """
        threshold = kwargs.get("zscore_threshold", 3)
        
        anomalies = pd.DataFrame(index=data.index)
        for column in data.columns:
            if column != "timestamp":
                # Remove minimum values
                column_data = self._remove_minimum(data[[column]])
                
                # Calculate z-scores for non-null values
                z_scores = pd.Series(index=data.index)
                non_null_data = column_data[column].dropna()
                
                if len(non_null_data) > 1:
                    z_scores[non_null_data.index] = stats.zscore(non_null_data)
                    
                    anomalies[f"{column}_is_anomaly"] = (z_scores > threshold) & (~z_scores.isna())
                else:
                    anomalies[f"{column}_is_anomaly"] = False

        anomaly_periods = self._identify_anomaly_periods(anomalies)
        return anomalies, anomaly_periods