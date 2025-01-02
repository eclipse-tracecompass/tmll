from dataclasses import dataclass
from enum import Enum, auto
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional, Tuple, cast
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, acf, pacf

from tmll.ml.modules.base_module import BaseModule
from tmll.common.models.experiment import Experiment
from tmll.common.models.output import Output
from tmll.tmll_client import TMLLClient
from tmll.ml.utils.document_generator import DocumentGenerator


class ResourceType(Enum):
    CPU = auto()
    MEMORY = auto()
    DISK = auto()


class ForecastMethod(Enum):
    ARIMA = auto()
    VAR = auto()
    MOVING_AVERAGE = auto()


@dataclass
class ResourceThresholds:
    cpu_threshold: float = 85.0  # percentage
    memory_threshold: float = 10 * 1024 * 1024  # MB
    disk_threshold: float = 50 * 1024 * 1024  # MB/s
    warning_period: str = '1m'  # time-based


@dataclass
class ResourceMetrics:
    current_usage: float
    peak_usage: float
    average_usage: float
    forecast_values: List[float]
    forecast_timestamps: List[pd.Timestamp]
    threshold_violations: List[Tuple[pd.Timestamp, pd.Timestamp, float, bool]]
    utilization_pattern: str
    units: str


@dataclass
class CapacityForecastResult:
    resource_metrics: Dict[str, ResourceMetrics]
    resource_type: ResourceType
    forecast_period: Dict[str, Optional[pd.Timestamp]]
    thresholds_used: ResourceThresholds
    forecast_method: ForecastMethod


class CapacityPlanning(BaseModule):
    """
    Capacity Planning Module

    This module analyzes system resource utilization to forecast future usage and provide
    capacity planning recommendations. It uses historical trace data to:
        1. Track current resource utilization patterns
        2. Forecast future resource usage
        3. Identify potential capacity bottlenecks
    """

    def __init__(self, client: TMLLClient, experiment: Experiment,
                 outputs: Optional[List[Output]] = None, **kwargs) -> None:
        """
        Initialize the capacity planning module.

        :param client: The TMLL client for data communication
        :type client: TMLLClient
        :param experiment: The experiment to analyze
        :type experiment: Experiment
        :param outputs: Optional list of outputs to analyze
        :type outputs: Optional[List[Output]]
        :param kwargs: Additional keyword arguments
        :type kwargs: dict
        """
        super().__init__(client, experiment)

        self.combined_df: Optional[pd.DataFrame] = None
        self.thresholds: ResourceThresholds = ResourceThresholds()
        self.scalers: Dict[str, StandardScaler] = {}

        self.logger.info("Initializing Capacity Planning module.")

        self._process(outputs, **kwargs)

    def _process(self, outputs: Optional[List[Output]] = None, **kwargs) -> None:
        if not outputs:
            outputs = self.experiment.find_outputs(keyword=["cpu usage", "memory usage", "disk"],
                                                   type=["xy"], match_any=True)

        super()._process(outputs=outputs,
                         normalize=False,
                         resample=True,
                         align_timestamps=True,
                         **kwargs)

    def _post_process(self, **kwargs) -> None:
        """Post-process the data after loading."""
        if not self.dataframes:
            return

        self.combined_df = self.data_preprocessor.combine_dataframes(list(self.dataframes.values()))

        for column in list(self.combined_df.columns):
            self.combined_df[f"{column}_original"] = self.combined_df[column]

            scaler = StandardScaler()
            self.combined_df[column] = scaler.fit_transform(np.array(self.combined_df[column]).reshape(-1, 1))
            self.scalers[column] = scaler

    def _format_bytes(self, bytes_value: float) -> Tuple[float, str]:
        """
        Convert bytes to human-readable format with appropriate unit.

        :param bytes_value: Value in bytes
        :type bytes_value: float
        :return: Tuple of (converted value, unit)
        :rtype: Tuple[float, str]
        """
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_idx = 0
        value = float(bytes_value)

        while value >= 1024 and unit_idx < len(units) - 1:
            value /= 1024
            unit_idx += 1

        return value, units[unit_idx]

    def _convert_time(self, time_in_seconds: float) -> str:
        """
        Convert seconds to a human-readable string with appropriate units.

        :param time: The time in seconds
        :type time: float
        :return: The time in human-readable format
        :rtype: str
        """
        units = ["s", "m", "h"]
        thresholds = [1, 60, 3600]
        time = abs(time_in_seconds)

        # If time is less than 1 second, convert to smaller units
        if time < 1:
            if time < 0.000001:  # nanoseconds
                return f"{time * 1e9:.2f} ns"
            elif time < 0.001:  # microseconds
                return f"{time * 1e6:.2f} us"
            else:  # milliseconds
                return f"{time * 1000:.2f} ms"

        # If time is greater than 1 second, convert to larger units
        for i in range(len(units) - 1, -1, -1):
            if time >= thresholds[i]:
                converted_time = time / thresholds[i]
                return f"{converted_time:.2f} {units[i]}"

        return f"{time:.2f} s"

    def _analyze_utilization_pattern(self, series: pd.Series) -> str:
        """
        Analyze the utilization pattern of a resource.

        :param series: Time series data of resource usage
        :type series: pd.Series
        :return: Description of the utilization pattern
        :rtype: str
        """
        # Calculate only for the 99th percentile of data to avoid outliers
        data = series[series < series.quantile(0.99)]

        mean = data.mean()
        std = data.std()
        cv = std / mean if mean > 0 else float("inf")

        if cv == float("inf"):
            return "No usage"

        if cv < 0.1:
            return "Very stable"
        elif cv < 0.3:
            return "Stable"
        elif cv < 0.6:
            return "Moderate variation"
        else:
            return "Highly variable"

    def _calculate_pdq(self, series: pd.Series, max_lag: int = 20) -> Tuple[int, int, int]:
        """
        Determine the optimal AR (p), I (d), and MA (q) parameters for an ARIMA model.

        :param series: Input time series data
        :type series: pd.Series
        :param max_lag: Maximum lag to consider for ACF and PACF
        :type max_lag: int
        :return: Optimal p, d, and q parameters
        :rtype: Tuple[int, int, int]
        """
        p, d, q = 0, 0, 0

        test_series = series.copy()
        while True:
            if test_series.nunique() == 1:
                break

            adf_result = adfuller(test_series)
            if adf_result[1] < 0.05:  # is stationary
                break

            # Detrend the series
            test_series = test_series.diff().dropna()
            d += 1

        series = series.diff(d).dropna() if d > 0 else series

        acf_values = acf(series, nlags=max_lag, fft=True)
        pacf_values = pacf(series, nlags=max_lag)

        # Determine p (PACF cutoff)
        p = next((lag for lag, value in enumerate(pacf_values) if abs(value) < 0.2), max_lag)

        # Determine q (ACF cutoff)
        q = next((lag for lag, value in enumerate(acf_values) if abs(value) < 0.2), max_lag)

        return p, d, q

    def _get_series_freq(self, series: pd.Series) -> str:
        """
        Get the frequency of a time series.

        :param series: Input time series data
        :type series: pd.Series
        :return: Frequency string
        :rtype: str
        """
        freq = pd.infer_freq(cast(pd.DatetimeIndex, series.index))
        if freq is None or not isinstance(freq, str):
            freq = "1s"
        if not freq[0].isdigit():
            freq = "1" + freq

        return freq

    def _parse_time_string(self, time_str: str) -> float:
        """
        Parse a time string into seconds.

        :param time_str: Time string (e.g., '1s', '500ms', '24h', '7d')
        :type time_str: str
        :return: Time in seconds
        :rtype: float
        :raises ValueError: If the time string format is invalid
        """
        pattern = r'^(\d+(?:\.\d+)?)(us|ms|s|m|h|d)$'
        match = re.match(pattern, time_str)

        if not match:
            return 1.0

        value, unit = float(match.group(1)), match.group(2)

        conversion = {
            'ns': 1e-9,
            'us': 1e-6,
            'ms': 1e-3,
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800
        }

        return value * conversion[unit]

    def _forecast_arima(self, series: pd.Series, forecast_steps: int) -> Tuple[List[float], List[pd.Timestamp]]:
        """
        Forecast future values using ARIMA model with dynamic parameters.

        :param series: Input time series data
        :type series: pd.Series
        :param forecast_steps: Number of steps to forecast
        :type forecast_steps: int
        :return: Tuple of forecast values and timestamps
        :rtype: Tuple[List[float], List[pd.Timestamp]]
        """
        freq = self._get_series_freq(series)
        p, d, q = self._calculate_pdq(series)

        try:
            model = ARIMA(series, order=(p, d, q))
            results = model.fit()

            forecast = results.forecast(steps=forecast_steps)

            last_timestamp = cast(pd.Timestamp, series.index[-1])
            future_timestamps = pd.date_range(
                start=last_timestamp + pd.Timedelta(freq),
                periods=forecast_steps,
                freq=freq
            )

            return forecast.tolist(), future_timestamps.tolist()
        except Exception as e:
            self.logger.error(f"ARIMA forecast failed")
            return [], []

    def _forecast_var(self, data: pd.DataFrame, forecast_steps: int) -> Dict[str, Tuple[List[float], List[pd.Timestamp]]]:
        """
        Forecast future values using VAR model.

        :param data: Input multivariate time series data
        :type data: pd.DataFrame
        :param forecast_steps: Number of steps to forecast
        :type forecast_steps: int
        :return: Dictionary of forecasts for each variable
        :rtype: Dict[str, Tuple[List[float], List[pd.Timestamp]]]
        """
        freq = self._get_series_freq(data.iloc[:, 0])

        model = VAR(data)
        results = model.fit()

        forecast = results.forecast(data.values, steps=forecast_steps)

        last_timestamp = cast(pd.Timestamp, data.index[-1])
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(freq),
            periods=forecast_steps,
            freq=freq
        )

        forecasts = {}
        for i, column in enumerate(data.columns):
            forecasts[column] = (forecast[:, i].tolist(), future_timestamps.tolist())

        return forecasts

    def _forecast_moving_average(self, series: pd.Series, forecast_steps: int, window_size: int) -> Tuple[List[float], List[pd.Timestamp]]:
        """
        Forecast future values using moving average method.

        :param series: Input time series data
        :type series: pd.Series
        :param forecast_steps: Number of steps to forecast
        :type forecast_steps: int
        :param window_size: Window size for moving average
        :type window_size: int
        :return: Tuple of forecast values and timestamps
        :rtype: Tuple[List[float], List[pd.Timestamp]]
        """
        freq = self._get_series_freq(series)

        forecast = series.rolling(window=window_size).mean().iloc[-forecast_steps:]
        last_timestamp = cast(pd.Timestamp, series.index[-1])
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(freq),
            periods=forecast_steps,
            freq=freq
        )

        return forecast.tolist(), future_timestamps.tolist()

    def _detect_threshold_violations(self, values: List[float], timestamps: List[pd.Timestamp],
                                     threshold: float, window_size: str) -> List[Tuple[pd.Timestamp, pd.Timestamp, float, bool]]:
        """
        Detect periods where the forecast exceeds the threshold.

        :param values: Forecasted values
        :type values: List[float]
        :param timestamps: Corresponding timestamps
        :type timestamps: List[pd.Timestamp]
        :param threshold: Threshold value
        :type threshold: float
        :param window_size: Window size for threshold detection
        :type window_size: float
        :return: List of threshold violation periods
        :rtype: List[Tuple[pd.Timestamp, float]]
        """
        window_seconds = self._parse_time_string(window_size)
        violations = []

        violation_start = None
        violation_max = float('-inf')

        for i, (value, timestamp) in enumerate(zip(values, timestamps)):
            if value > threshold:
                # Start or continue a violation period
                if violation_start is None:
                    violation_start = timestamp
                violation_max = max(violation_max, value)

                # If this is the last point or next point doesn't exceed threshold, end the current violation period
                is_last_point = i == len(values) - 1
                next_point_violates = (not is_last_point and values[i + 1] > threshold)

                if is_last_point or not next_point_violates:
                    violation_duration = (timestamp - violation_start).total_seconds()
                    is_sustained = violation_duration >= window_seconds

                    violations.append((
                        violation_start,
                        timestamp,
                        violation_max,
                        is_sustained
                    ))

                    violation_start = None
                    violation_max = float('-inf')

        return violations

    def forecast_capacity(self, resource_types: Optional[List[ResourceType]] = None,
                          method: str = ForecastMethod.ARIMA.name,
                          forecast_steps: int = 100,
                          **kwargs) -> Dict[ResourceType, CapacityForecastResult]:
        """
        Forecast future resource utilization and identify potential capacity issues.

        :param resource_types: List of resource types to analyze
        :type resource_types: Optional[List[ResourceType]]
        :param method: Forecasting method to use (e.g., ARIMA, VAR, Moving Average)
        :type method: ForecastMethod
        :param forecast_steps: Number of steps to forecast
        :type forecast_steps: int
        :return: Forecast results for each resource type
        :rtype: Dict[ResourceType, CapacityForecastResult]
        """
        if self.combined_df is None or self.combined_df.empty:
            self.logger.warning("No data available for capacity planning analysis")
            return {}

        # Check if method is present in the forecast methods
        if not hasattr(ForecastMethod, method.upper()):
            self.logger.warning(f"Invalid forecast method: {method}. Using default ARIMA.")
            method = ForecastMethod.ARIMA.name
        method = method.upper()

        if resource_types is None:
            resource_types = list(ResourceType)

        self.thresholds = ResourceThresholds(
            cpu_threshold=kwargs.get("cpu_threshold", ResourceThresholds.cpu_threshold),
            memory_threshold=kwargs.get("memory_threshold", ResourceThresholds.memory_threshold),
            disk_threshold=kwargs.get("disk_threshold", ResourceThresholds.disk_threshold),
            warning_period=kwargs.get("warning_period", ResourceThresholds.warning_period)
        )

        results = {}

        all_resource_columns = []
        resource_columns_by_type = {}
        for resource_type in resource_types:
            columns = [
                col for col in self.combined_df.columns
                if (
                    (resource_type == ResourceType.CPU and "cpu" in col.lower()) or
                    (resource_type == ResourceType.MEMORY and "memory usage" in col.lower()) or
                    (resource_type == ResourceType.DISK and "disk" in col.lower())
                ) and not col.endswith("_original")
            ]
            if columns:
                all_resource_columns.extend(columns)
                resource_columns_by_type[resource_type] = columns
            else:
                self.logger.warning(f"No {resource_type.name} resources found in data")

        if not all_resource_columns:
            self.logger.warning("No resources found in data")
            return {}

        # Generate forecasts based on method
        resource_metrics = {}
        if method in [ForecastMethod.ARIMA.name, ForecastMethod.MOVING_AVERAGE.name]:
            # Generate individual ARIMA or Moving Average forecasts for each column
            for name in all_resource_columns:
                series = self.combined_df[name]
                if method == ForecastMethod.ARIMA.name:
                    forecast_values, forecast_timestamps = self._forecast_arima(series, forecast_steps)
                else:
                    window_size = kwargs.get("window_size", 5)
                    forecast_values, forecast_timestamps = self._forecast_moving_average(series, forecast_steps, window_size)
                resource_metrics[name] = (forecast_values, forecast_timestamps)
        elif method == ForecastMethod.VAR.name:
            # Generate VAR forecasts for all columns at once
            data = self.combined_df[all_resource_columns]
            var_forecasts = self._forecast_var(data, forecast_steps)
            resource_metrics = var_forecasts

        for resource_type, columns in resource_columns_by_type.items():
            if resource_type == ResourceType.CPU:
                threshold = self.thresholds.cpu_threshold
                units = "%"
            elif resource_type == ResourceType.MEMORY:
                threshold = self.thresholds.memory_threshold
                units = "bytes"
            else:
                threshold = self.thresholds.disk_threshold
                units = "bytes/s"

            type_metrics = {}
            forecast_timestamps = []
            for name in columns:
                original_series = self.combined_df[f"{name}_original"]
                forecast_values, forecast_timestamps = resource_metrics[name]

                forecast_values = self.scalers[name].inverse_transform(np.array(forecast_values)
                                                                       .reshape(-1, 1)).flatten().tolist()

                violations = self._detect_threshold_violations(
                    forecast_values,
                    forecast_timestamps,
                    threshold,
                    self.thresholds.warning_period
                )

                type_metrics[name] = ResourceMetrics(
                    current_usage=original_series.iloc[-1],
                    peak_usage=original_series.max(),
                    average_usage=original_series.mean(),
                    forecast_values=forecast_values,
                    forecast_timestamps=forecast_timestamps,
                    threshold_violations=violations,
                    utilization_pattern=self._analyze_utilization_pattern(original_series),
                    units=units
                )

            results[resource_type] = CapacityForecastResult(
                resource_metrics=type_metrics,
                resource_type=resource_type,
                forecast_period={
                    "start": forecast_timestamps[0],
                    "end": forecast_timestamps[-1]
                },
                thresholds_used=self.thresholds,
                forecast_method=ForecastMethod[method.upper()]
            )

        return results

    def interpret(self, forecast_results: Dict[ResourceType, CapacityForecastResult]) -> None:
        """
        Interpret capacity planning analysis results and provide recommendations.

        :param forecast_results: Results from capacity forecast analysis
        :type forecast_results: Dict[ResourceType, CapacityForecastResult]
        """
        DocumentGenerator.section("Capacity Planning Analysis Results")

        for resource_type, result in forecast_results.items():
            DocumentGenerator.section(f"{resource_type.name} Capacity Analysis")

            parameters = {
                "Analysis Period Start": result.forecast_period["start"].strftime("%Y-%m-%d %H:%M:%S.%f") if result.forecast_period["start"] else "N/A",
                "Analysis Period End": result.forecast_period["end"].strftime("%Y-%m-%d %H:%M:%S.%f") if result.forecast_period["end"] else "N/A",
                "Forecast Method": result.forecast_method.name,
            }

            if resource_type.name == ResourceType.CPU.name:
                parameters["CPU Threshold"] = f"{result.thresholds_used.cpu_threshold}%"
            elif resource_type.name == ResourceType.MEMORY.name:
                val, unit = self._format_bytes(result.thresholds_used.memory_threshold)
                parameters["Memory Threshold"] = f"{val:.2f} {unit}"
            else:
                val, unit = self._format_bytes(result.thresholds_used.disk_threshold)
                parameters["Disk Threshold"] = f"{val:.2f} {unit}/s"

            DocumentGenerator.metrics_group("Analysis Parameters", parameters)

            # Resource metrics
            for resource_name, metrics in result.resource_metrics.items():
                resource_metrics = {
                    "Current Usage": f"{metrics.current_usage:.1f}{metrics.units}",
                    "Peak Usage": f"{metrics.peak_usage:.1f}{metrics.units}",
                    "Average Usage": f"{metrics.average_usage:.1f}{metrics.units}",
                    "Utilization Pattern": metrics.utilization_pattern
                }

                if resource_type.name != ResourceType.CPU.name:
                    for key in ["Current Usage", "Peak Usage", "Average Usage"]:
                        value, unit = self._format_bytes(metrics.__dict__[key.lower().replace(" ", "_")])
                        resource_metrics[key] = f"{value:.2f} {unit}"
                        if resource_type.name == ResourceType.DISK.name:
                            resource_metrics[key] += "/s"

                if metrics.threshold_violations:
                    next_violation = metrics.threshold_violations[0]
                    resource_metrics["Next Threshold Violation"] = (
                        f"{next_violation[0].strftime("%Y-%m-%d %H:%M:%S.%f")} "
                    )
                    resource_metrics["Total Violations"] = str(len(metrics.threshold_violations))

                DocumentGenerator.metrics_group(f"Resource: {resource_name}", resource_metrics)

                if metrics.threshold_violations:
                    violation_headers = ["Start", "End", "Duration", "Forecasted Usage"]
                    num_sig_violations = min(5, len(metrics.threshold_violations))
                    violation_rows = []
                    for i in range(num_sig_violations):
                        start_time, end_time, max_usage, _ = metrics.threshold_violations[i]
                        duration = self._convert_time((end_time - start_time).total_seconds())
                        if resource_type.name == ResourceType.CPU.name:
                            max_usage = f"{max_usage:.2f}%"
                        else:
                            max_value, max_unit = self._format_bytes(max_usage)
                            max_usage = f"{max_value:.2f} {max_unit}"
                            if resource_type.name == ResourceType.DISK.name:
                                max_usage += "/s"

                        violation_rows.append([start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                               end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                               duration, max_usage])
                    DocumentGenerator.table(
                        violation_headers,
                        violation_rows,
                        f"Top Threshold Violations for {resource_name}"
                    )

        # Generate recommendations
        DocumentGenerator.section("Capacity Planning Recommendations")

        DEFAULT_RECOMMENDATION = "Resource utilization is within acceptable limits. No immediate action required."
        recommendations = {
            "Immediate Actions": [],
            "Short-term Planning": [],
            "Long-term Strategy": [],
            "Optimization Opportunities": []
        }

        for resource_type, result in forecast_results.items():
            for resource_name, metrics in result.resource_metrics.items():
                average_usage = metrics.average_usage
                peak_usage = metrics.peak_usage
                pattern = metrics.utilization_pattern
                violations = metrics.threshold_violations

                threshold = (
                    result.thresholds_used.cpu_threshold if resource_type.name == ResourceType.CPU.name
                    else result.thresholds_used.memory_threshold if resource_type.name == ResourceType.MEMORY.name
                    else result.thresholds_used.disk_threshold
                )

                if violations:
                    if self.combined_df is None or self.combined_df.empty:
                        continue

                    time_to_violation = (violations[0][0] - self.combined_df.index[-1]).total_seconds()

                    # Immediate actions for imminent violations
                    if time_to_violation < 3600:  # Less than 1 hour
                        time_to_violation = self._convert_time(time_to_violation)
                        if resource_type.name == ResourceType.CPU.name:
                            recommendations["Immediate Actions"].append(
                                f"Critical: {resource_name} will exceed {threshold}% in {time_to_violation}. "
                                f"Consider immediate load balancing or scaling up CPU capacity."
                            )
                        elif resource_type.name == ResourceType.MEMORY.name:
                            recommendations["Immediate Actions"].append(
                                f"Critical: {resource_name} will exceed {threshold}% in {time_to_violation} "
                                f"Consider freeing up memory or increasing available memory."
                            )
                        else:
                            recommendations["Immediate Actions"].append(
                                f"Critical: {resource_name} will exceed {threshold}% in {time_to_violation}. "
                                f"Consider cleanup or adding storage capacity."
                            )

                    # Short-term planning for upcoming violations
                    elif time_to_violation < 86400:  # Less than 24 hours
                        time_to_violation = self._convert_time(time_to_violation)
                        if resource_type.name == ResourceType.CPU.name:
                            recommendations["Short-term Planning"].append(
                                f"{resource_name} will exceed {threshold}% in {time_to_violation} "
                                f"Plan for CPU capacity increase or workload redistribution."
                            )
                        elif resource_type.name == ResourceType.MEMORY.name:
                            recommendations["Short-term Planning"].append(
                                f"{resource_name} will exceed {threshold}% in {time_to_violation} "
                                f"Plan for memory upgrade or optimization."
                            )
                        else:
                            recommendations["Short-term Planning"].append(
                                f"{resource_name} will exceed {threshold}% in {time_to_violation}. "
                                f"Plan for storage expansion or archival."
                            )

                # Long-term strategy based on utilization patterns
                if pattern == "Highly variable":
                    recommendations["Long-term Strategy"].append(
                        f"{resource_name} shows highly variable usage patterns. "
                        f"Consider implementing auto-scaling or dynamic resource allocation."
                    )

                # Long-term strategy based on peak usage
                if peak_usage > threshold * 0.9:
                    if resource_type.name == ResourceType.CPU.name:
                        peak_usage_str = f"{peak_usage:.1f}%"
                        threshold_str = f"{threshold}%"
                    else:
                        peak_value, peak_unit = self._format_bytes(peak_usage)
                        threshold_value, threshold_unit = self._format_bytes(threshold)
                        peak_usage_str = f"{peak_value:.1f} {peak_unit}"
                        threshold_str = f"{threshold_value:.1f} {threshold_unit}"
                        if resource_type.name == ResourceType.DISK.name:
                            peak_usage_str += "/s"
                            threshold_str += "/s"

                    recommendations["Long-term Strategy"].append(
                        f"{resource_name} has peak usage ({peak_usage_str}) approaching or bypassing the threshold ({threshold_str}). "
                        f"Consider increasing capacity or implementing load balancing."
                    )

                # Optimization opportunities
                if pattern == "Stable" and average_usage < threshold * 0.3:  # Less than 30% of threshold
                    recommendations["Optimization Opportunities"].append(
                        f"{resource_name} is consistently underutilized. "
                        f"Consider resource consolidation or downsizing."
                    )

        for category, items in recommendations.items():
            if not items:
                items = [DEFAULT_RECOMMENDATION]
            items = {f"{idx + 1}": item for idx, item in enumerate(items)}
            DocumentGenerator.metrics_group(category, items)

    def plot_capacity_forecast(self, forecast_results: Dict[ResourceType, CapacityForecastResult],
                               zoomed: bool = False, **kwargs) -> None:
        """
        Plot capacity forecast analysis results.

        :param forecast_results: Results from capacity forecast analysis
        :type forecast_results: Dict[ResourceType, CapacityForecastResult]
        :param zoomed: Whether to zoom in on the forecast period (show a smaller time range of original data)
        :type zoomed: bool
        :param kwargs: Additional keyword arguments for plotting
        :type kwargs: dict
        """
        if not forecast_results:
            self.logger.warning("No forecast results to plot")
            return

        if self.combined_df is None or self.combined_df.empty:
            self.logger.warning("No data available for plotting")
            return

        fig_size = kwargs.get("fig_size", (15, 5))
        fig_dpi = kwargs.get("fig_dpi", 100)
        colors = plt.get_cmap("tab20")

        for resource_type, result in forecast_results.items():
            for resource_name, metrics in result.resource_metrics.items():
                historical_data = self.combined_df[f"{resource_name}_original"]

                # Get the last 10% of the data for zoomed view
                if zoomed:
                    historical_data = historical_data.iloc[-int(len(historical_data) * 0.1):]

                forecast_data = pd.Series(
                    metrics.forecast_values,
                    index=metrics.forecast_timestamps,
                    name=resource_name
                )

                plots = []

                # Plot historical data
                plots.append({
                    "plot_type": "time_series",
                    "data": historical_data,
                    "label": "Historical",
                    "color": colors(0),
                    "alpha": 0.8,
                    "linewidth": 2
                })

                # Plot forecast
                plots.append({
                    "plot_type": "time_series",
                    "data": forecast_data,
                    "label": "Forecast",
                    "color": colors(1),
                    "alpha": 0.8,
                    "linewidth": 2,
                    "linestyle": "--"
                })

                # Add threshold line
                if resource_type.name == ResourceType.CPU.name:
                    threshold_line = result.thresholds_used.cpu_threshold
                elif resource_type.name == ResourceType.MEMORY.name:
                    threshold_line = result.thresholds_used.memory_threshold
                else:
                    threshold_line = result.thresholds_used.disk_threshold

                units = metrics.units
                threshold = threshold_line
                if resource_type.name != ResourceType.CPU.name:
                    threshold_value, threshold_unit = self._format_bytes(threshold)
                    threshold = round(threshold_value, 2)
                    units = threshold_unit

                    if resource_type.name == ResourceType.DISK.name:
                        units += "/s"

                plots.append({
                    "plot_type": "hline",
                    "y": threshold_line,
                    "color": "red",
                    "linestyle": "--",
                    "label": f"Threshold ({threshold} {units})",
                    "alpha": 0.5
                })

                # Highlight threshold violations
                for violation_start, violation_end, _, is_sustained in metrics.threshold_violations:
                    if not is_sustained:
                        continue

                    plots.append({
                        "plot_type": "span",
                        "start": violation_start,
                        "end": violation_end,
                        "color": "red",
                        "alpha": 0.2
                    })

                if resource_type.name == ResourceType.CPU.name:
                    usage_str = f"{metrics.average_usage:.1f}%"
                    peak_str = f"{metrics.peak_usage:.1f}%"
                else:
                    avg_val, avg_unit = self._format_bytes(metrics.average_usage)
                    peak_val, peak_unit = self._format_bytes(metrics.peak_usage)
                    usage_str = f"{avg_val:.1f} {avg_unit}"
                    peak_str = f"{peak_val:.1f} {peak_unit}"
                    if resource_type.name == ResourceType.DISK.name:
                        usage_str += "/s"
                        peak_str += "/s"

                title = (
                    f"{resource_type.name} Capacity Forecast: {resource_name}\n"
                    f"(Average Usage: {usage_str}, "
                    f"Peak: {peak_str}, "
                    f"Pattern: {metrics.utilization_pattern})"
                )

                max_value = max(historical_data.max(), forecast_data.max(), threshold_line)
                y_ticks = np.linspace(0, max_value, 10)
                if resource_type.name == ResourceType.CPU.name:
                    y_tick_labels = [f"{tick:.0f}%" for tick in y_ticks]
                elif resource_type.name == ResourceType.MEMORY.name:
                    y_tick_labels = [f"{self._format_bytes(tick)[0]:.0f} {self._format_bytes(tick)[1]}" for tick in y_ticks]
                else:
                    y_tick_labels = [f"{self._format_bytes(tick)[0]:.0f} {self._format_bytes(tick)[1]}/s" for tick in y_ticks]

                self._plot(
                    plots,
                    plot_size=fig_size,
                    dpi=fig_dpi,
                    grid=True,
                    fig_title=title,
                    fig_xlabel="Time",
                    fig_ylabel=f"Usage",
                    fig_yticks=y_ticks,
                    fig_yticklabels=y_tick_labels
                )
