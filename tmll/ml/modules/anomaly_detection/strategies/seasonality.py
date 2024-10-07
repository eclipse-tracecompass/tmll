from statsmodels.tsa.seasonal import STL
from scipy import stats
import pandas as pd
import numpy as np
from typing import List, Tuple

from tmll.ml.modules.anomaly_detection.strategies.base import AnomalyDetectionStrategy

class SeasonalityStrategy(AnomalyDetectionStrategy):

    def detect_anomalies(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Detect anomalies using seasonality analysis via STL decomposition.

        :param data: The data to analyze for anomalies
        :type data: pd.DataFrame
        :param kwargs: Additional parameters for the seasonality method
        :type kwargs: dict
        :return: DataFrame with detected anomalies and list of anomaly periods
        :rtype: Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]
        """
        period = kwargs.get('seasonal_period', None)
        anomalies_list = []
        
        # Check if data has a datetime index
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            self.logger.error('Data must have a datetime index.')
            return pd.DataFrame(index=data.index, columns=data.columns).fillna(False), []

        for column in data.columns:
            # Skip non-numeric columns
            if data[column].dtype.kind in 'bifc':
                # Remove minimum values
                column_data = self._remove_minimum(data[[column]])
                ts = column_data[column].dropna()

                # Resample data to a fixed frequency if necessary
                if ts.index.inferred_freq is None: # type: ignore
                    # Resample data to a fixed frequency
                    resample_freq = kwargs.get('resample_freq', None)
                    if resample_freq is None:
                        inferred_freq = pd.infer_freq(ts.index) # type: ignore
                        if inferred_freq is None:
                            self.logger.error('Cannot infer frequency of data. Please specify resample_freq.')
                            return pd.DataFrame(index=data.index, columns=data.columns).fillna(False), []
                        else:
                            resample_freq = inferred_freq
                    ts = ts.resample(resample_freq).mean()
                else:
                    resample_freq = ts.index.inferred_freq  # type: ignore

                # Determine the period
                if period is None:
                    freq_map = {'d': 7, 'h': 24, 'min': 60, 's': 60, 'ms': 1000}
                    period = freq_map.get(resample_freq[0], None)
                    if period is None:
                        period = kwargs.get('default_period', None)
                        if period is None:
                            self.logger.error('Cannot determine period of data. Please specify seasonal_period.')
                            return pd.DataFrame(index=data.index, columns=data.columns).fillna(False), []

                try:
                    stl = STL(ts, period=period, robust=True)
                    result = stl.fit()
                    residual = result.resid
                except:
                    continue

                # Detect anomalies in residuals using Z-score
                threshold = kwargs.get('zscore_threshold', 3)
                z_scores = np.abs(stats.zscore(residual.dropna()))
                anomalies_col = pd.Series(False, index=ts.index, name=f'{column}_is_anomaly')
                anomalies_col.loc[residual.dropna().index] = z_scores > threshold
                anomalies_list.append(anomalies_col)
            else:
                # For non-numeric columns, create a Series of False
                anomalies_col = pd.Series(False, index=data.index, name=f'{column}_is_anomaly')
                anomalies_list.append(anomalies_col)

        # Combine all anomaly Series into a DataFrame
        anomalies = pd.concat(anomalies_list, axis=1).reindex(data.index, fill_value=False)

        anomaly_periods = self._identify_anomaly_periods(anomalies)
        return anomalies, anomaly_periods
