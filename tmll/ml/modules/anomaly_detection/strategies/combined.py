import pandas as pd
from typing import Tuple
from typing import List

from tmll.ml.modules.anomaly_detection.strategies.base import AnomalyDetectionStrategy

class CombinedStrategy(AnomalyDetectionStrategy):
    def __init__(self, strategies: List[AnomalyDetectionStrategy]):
        self.strategies = strategies

    def detect_anomalies(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Detect anomalies by combining multiple detection methods.

        :param data: The data to analyze for anomalies
        :type data: pd.DataFrame
        :param kwargs: Additional parameters for each detection method
        :type kwargs: dict
        :return: DataFrame with combined anomalies
        :rtype: pd.DataFrame
        """
        all_anomalies = None
        for strategy in self.strategies:
            anomalies, _ = strategy.detect_anomalies(data, **kwargs)
            if all_anomalies is None:
                all_anomalies = anomalies
            else:
                all_anomalies &= anomalies

        if all_anomalies is None:
            all_anomalies = pd.DataFrame(index=data.index, columns=data.columns).fillna(False)

        anomaly_periods = self._identify_anomaly_periods(all_anomalies)
        return all_anomalies, anomaly_periods