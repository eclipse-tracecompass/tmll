import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

from tmll.ml.modules.anomaly_detection.strategies.base import AnomalyDetectionStrategy

class FrequencyDomainStrategy(AnomalyDetectionStrategy):
    def detect_anomalies(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        window_size = kwargs.get("window_size", 512)
        overlap = kwargs.get("overlap", 0.5)
        threshold = kwargs.get("threshold", 3)  # Threshold in terms of MADs

        # Keep a copy of the original data
        df = data.copy()

        # Ensure the index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        anomalies = pd.DataFrame(index=df.index)

        for column in df.columns:
            if df[column].dtype.kind in 'bifc':  # Check if column is numeric
                column_data = self._remove_minimum(df[[column]])
                np_data = column_data[column].values

                # Adjust window size if it's larger than half the data size
                if window_size > len(np_data) // 2:
                    window_size = len(np_data) // 2
                    if window_size % 2 != 0:
                        window_size -= 1  # Ensure window_size is even

                # Ensure we have enough data for windowing
                if len(np_data) < window_size or window_size <= 0:
                    anomalies[f"{column}_is_anomaly"] = False
                    anomalies[f"{column}_anomaly_score"] = 0
                    continue

                fft_results, window_indices = self._windowed_fft(np.array(np_data), window_size, overlap)

                # Ensure we have valid FFT results
                if not fft_results:
                    anomalies[f"{column}_is_anomaly"] = False
                    anomalies[f"{column}_anomaly_score"] = 0
                    continue

                anomaly_scores = self._calculate_anomaly_scores(fft_results)

                # Initialize a Series to accumulate anomaly scores
                scores_series = pd.Series(0, index=df.index, dtype=float)
                counts_series = pd.Series(0, index=df.index, dtype=int)

                # Accumulate anomaly scores for overlapping windows
                for idx_range, score in zip(window_indices, anomaly_scores):
                    start_idx, end_idx = idx_range
                    scores_series.iloc[start_idx:end_idx] += score
                    counts_series.iloc[start_idx:end_idx] += 1

                # Avoid division by zero
                counts_series = counts_series.replace(0, np.nan)
                average_scores = scores_series / counts_series

                if average_scores.isna().all():
                    anomalies[f"{column}_is_anomaly"] = False
                    anomalies[f"{column}_anomaly_score"] = 0
                    continue

                # Smooth the anomaly scores using a rolling mean
                smoothed_anomaly_scores = average_scores.rolling(window=5, min_periods=1).mean()

                # Detect anomalies using median and MAD
                median_score = smoothed_anomaly_scores.median()
                mad_score = np.median(np.abs(smoothed_anomaly_scores - median_score))
                anomaly_threshold = median_score + threshold * mad_score

                is_anomaly = smoothed_anomaly_scores > anomaly_threshold
                anomalies[f"{column}_is_anomaly"] = is_anomaly.fillna(False)
                anomalies[f"{column}_anomaly_score"] = smoothed_anomaly_scores.fillna(0)
            else:
                # For non-numeric columns, create a Series of False
                anomalies[f"{column}_is_anomaly"] = False
                anomalies[f"{column}_anomaly_score"] = 0

        anomaly_periods = self._identify_anomaly_periods(anomalies.filter(regex="_is_anomaly$"))
        return anomalies, anomaly_periods

    def _windowed_fft(self, data: np.ndarray, window_size: int, overlap: float) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Perform windowed FFT on the input data, ignoring windows with NaN values.
        Returns a list of FFT results and their corresponding index ranges.
        """
        step = int(window_size * (1 - overlap))
        if step <= 0:
            return [], []

        fft_results = []
        window_indices = []
        for i in range(0, len(data) - window_size + 1, step):
            window = data[i:i + window_size]

            # Check if the window contains NaNs, if so skip this window
            if np.isnan(window).any():
                continue

            # Perform FFT and append result
            fft_result = np.abs(np.fft.fft(window))
            fft_results.append(fft_result)
            window_indices.append((i, i + window_size))

        return fft_results, window_indices

    def _calculate_anomaly_scores(self, fft_results: List[np.ndarray]) -> np.ndarray:
        """
        Calculate anomaly scores based on FFT results.
        """
        fft_array = np.array(fft_results)
        if fft_array.size == 0:
            return np.array([])

        # Normalize each FFT result
        fft_norms = np.linalg.norm(fft_array, axis=1, keepdims=True)
        normalized_fft = fft_array / fft_norms

        # Calculate the median spectrum
        median_spectrum = np.median(normalized_fft, axis=0)

        # Calculate the Euclidean distance between each normalized FFT result and the median spectrum
        differences = np.linalg.norm(normalized_fft - median_spectrum, axis=1)

        return differences
