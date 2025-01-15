import pandas as pd
import numpy as np
import warnings
from typing import List, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from tmll.ml.modules.anomaly_detection.strategies.base import AnomalyDetectionStrategy

class SeasonalityStrategy(AnomalyDetectionStrategy):

    def detect_anomalies(self, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Detect anomalies using ARIMA prediction model.

        :param data: The data to analyze for anomalies
        :type data: pd.DataFrame
        :param kwargs: Additional parameters for the ARIMA method
        :type kwargs: dict
        :return: DataFrame with detected anomalies and list of anomaly periods
        :rtype: Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]
        """
        anomalies_list = []
        
        # Check if data has a datetime index
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            self.logger.error("Data must have a datetime index.")
            return pd.DataFrame(index=data.index, columns=data.columns), []
        
        resample_freq, seasonal_period = self._estimate_parameters(data)
        if kwargs.get("resample_freq", None) is not None:
            resample_freq = kwargs["resample_freq"]
            del kwargs["resample_freq"]
        if kwargs.get("seasonality_seasonal_period", None) is not None:
            seasonal_period = kwargs["seasonality_seasonal_period"]
            del kwargs["seasonality_seasonal_period"]

        for column in data.columns:
            # Skip non-numeric columns
            if data[column].dtype.kind in "bifc":
                # Remove minimum values
                column_data = self._remove_minimum(data[[column]])
                ts = column_data[column].dropna()

                # Fit ARIMA model and detect anomalies
                try:
                    anomalies_col, predicted_mean, conf_int = self._detect_anomalies_arima(ts, resample_freq, seasonal_period, **kwargs)
                except Exception as e:
                    self.logger.error(f"Error detecting anomalies for column {column}: {str(e)}")
                    anomalies_col = pd.Series(False, index=data.index, name=f"{column}_is_anomaly")
                
                anomalies_list.append(anomalies_col)
            else:
                # For non-numeric columns, create a Series of False
                anomalies_col = pd.Series(False, index=data.index, name=f"{column}_is_anomaly")
                anomalies_list.append(anomalies_col)
        
        # Combine all anomaly Series into a DataFrame
        anomalies = pd.concat(anomalies_list, axis=1)

        anomaly_periods = self._identify_anomaly_periods(anomalies)
        return anomalies, anomaly_periods

    def _detect_anomalies_arima(self, ts: pd.Series, resample_freq: str, seasonal_period: int, **kwargs) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Detect anomalies using ARIMA model predictions.

        :param ts: Time series data
        :param resample_freq: Resampling frequency
        :param seasonal_period: Seasonal period for SARIMA model
        :param kwargs: Additional parameters
        :return: Tuple of (anomalies series, predicted mean series, confidence interval DataFrame)
        """
        # Clean the data
        ts = self._clean_timeseries(ts, resample_freq)
        
        if len(ts) < 2:
            raise ValueError("Not enough valid data points after cleaning")

        # Determine if the series is stationary
        try:
            adf_result = adfuller(ts.values)
            is_stationary = adf_result[1] <= 0.05
        except Exception as e:
            self.logger.warning(f"ADF test failed: {str(e)}. Assuming non-stationary series.")
            is_stationary = False

        # Set ARIMA parameters
        p, d, q = kwargs.get("seasonality_arima_order", (1, int(not is_stationary), 1))
        P, D, Q = kwargs.get("seasonality_seasonal_order", (1, 1, 1))
        
        # Fit SARIMA model
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period))
                results = model.fit(disp=False, maxiter=1000)
        except Exception as e:
            self.logger.error(f"SARIMA model fitting failed: {str(e)}")
            return pd.Series(False, index=ts.index, name=f"{ts.name}_is_anomaly"), pd.Series(), pd.DataFrame()

        # Make predictions
        predictions = results.get_prediction(start=ts.index[0], end=ts.index[-1]) # type: ignore
        predicted_mean = predictions.predicted_mean
        
        # Calculate confidence intervals
        conf_int = predictions.conf_int(alpha=kwargs.get("seasonality_confidence_level", 0.05))

        # Detect anomalies
        anomalies = (ts < conf_int.iloc[:, 0]) | (ts > conf_int.iloc[:, 1])
        
        return pd.Series(anomalies, index=ts.index, name=f"{ts.name}_is_anomaly"), predicted_mean, conf_int

    def _clean_timeseries(self, ts: pd.Series, resample_freq: str) -> pd.Series:
        """
        Clean the time series by removing NaN and inf values, and interpolating missing data.

        :param ts: Input time series
        :param resample_freq: Resampling frequency
        :return: Cleaned time series
        """
        # Remove inf values
        ts = ts.replace([np.inf, -np.inf], np.nan)
        
        # Set a consistent frequency
        ts = ts.asfreq(resample_freq)
        
        # Interpolate missing values
        ts = ts.interpolate(method="time")
        
        # Remove any remaining NaN values at the beginning or end of the series
        ts = ts.dropna()
        
        return ts
    
    def _estimate_parameters(self, data: pd.DataFrame) -> Tuple[str, int]:
        """
        Estimate appropriate parameters for the given dataset.
        
        :param data: The input DataFrame with a datetime index
        :return: A tuple with estimated resampling frequency and seasonal period
        """
        # Calculate the total duration of the dataset
        duration = data.index[-1] - data.index[0]
        
        # Estimate resampling frequency
        total_points = len(data)
        avg_interval = duration / total_points
        
        if avg_interval.total_seconds() < 0.001:
            resample_freq = "ms"
        elif avg_interval.total_seconds() < 0.01:
            resample_freq = "10ms"
        elif avg_interval.total_seconds() < 0.1:
            resample_freq = "100ms"
        elif avg_interval.total_seconds() < 1:
            resample_freq = "s"  # 1 second
        elif avg_interval.total_seconds() < 60:
            resample_freq = "10s"  # 10 seconds
        else:
            resample_freq = "1min"  # 1 minute
        
        # Estimate seasonal period
        if duration.total_seconds() < 600:
            seasonal_period = 10 # 10 second cycle (10 points if resampled to seconds)
        elif duration.total_seconds() < 3600:  # Less than 1 hour
            seasonal_period = 60  # 1 minute cycle (60 points if resampled to seconds)
        elif duration.total_seconds() < 14400:  # Less than 4 hours
            seasonal_period = 300  # 5 minute cycle (300 points if resampled to seconds)
        else:
            seasonal_period = 900  # 15 minute cycle (900 points if resampled to seconds)
        
        return resample_freq, seasonal_period