import pandas as pd
import numpy as np
from typing import List, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from tmll.ml.modules.anomaly_detection.strategies.base import AnomalyDetectionStrategy

class IsolationForestStrategy(AnomalyDetectionStrategy):
    def detect_anomalies(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Detect anomalies using the Isolation Forest method.

        :param data: The data to analyze for anomalies
        :type data: pd.DataFrame
        :param kwargs: Additional parameters for the Isolation Forest method
        :type kwargs: dict
        :return: DataFrame with detected anomalies
        :rtype: pd.DataFrame
        """
        window_size = kwargs.get("iforest_window_size", 100)
        contamination = kwargs.get("iforest_contamination", 0.1)
        random_state = kwargs.get("iforest_random_state", 42)

        # Keep a copy of the original data
        df = data.copy()

        X = self._prepare_features(df, window_size)
        if X.shape[0] == 0:
            return pd.DataFrame(index=df.index, columns=df.columns), []
        
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        anomaly_scores = -iso_forest.fit_predict(X)

        scaler = StandardScaler()
        normalized_scores = scaler.fit_transform(anomaly_scores.reshape(-1, 1)).flatten()

        threshold = np.percentile(normalized_scores, 100 * (1 - contamination))
        anomalies = normalized_scores > threshold

        if not np.any(anomalies):
            threshold = np.percentile(normalized_scores, 99)
            anomalies = normalized_scores > threshold

        result_df = df.iloc[window_size-1:].copy()
        result_df["anomaly_score"] = normalized_scores
        for column in df.columns:
            if column != "timestamp":
                result_df[f"{column}_is_anomaly"] = anomalies

        anomaly_periods = self._identify_anomaly_periods(result_df.filter(regex="_is_anomaly$"))

        return result_df, anomaly_periods

    def _prepare_features(self, df: pd.DataFrame, window_size: int) -> np.ndarray:
        """
        Prepare features for Isolation Forest.

        :param df: The DataFrame to prepare features from
        :type df: pd.DataFrame
        :param window_size: The size of the rolling window for feature calculation
        :type window_size: int
        :return: Prepared feature matrix
        :rtype: np.ndarray
        """
        # Convert timestamp to datetime if it"s not already
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        
        # Convert timestamp to seconds from start of trace
        df["seconds_from_start"] = (df.index - df.index.min()).total_seconds()
        
        # Calculate rolling statistics
        for col in df.columns:
            if col not in ["timestamp", "seconds_from_start"]:
                df[f"{col}_rolling_mean"] = df[col].rolling(window=window_size).mean()
                df[f"{col}_rolling_std"] = df[col].rolling(window=window_size).std()
                df[f"{col}_rate_of_change"] = df[col].diff() / df["seconds_from_start"].diff()
        
        # Drop NaN values resulting from rolling calculations
        df = df.dropna()
        
        return df.values