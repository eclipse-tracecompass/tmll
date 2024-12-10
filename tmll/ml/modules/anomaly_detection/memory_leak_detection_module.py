import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Tuple, Dict, cast
from scipy import stats
from dataclasses import dataclass
from enum import Enum, auto

from tmll.ml.modules.base_module import BaseModule
from tmll.common.models.experiment import Experiment
from tmll.common.models.output import Output
from tmll.tmll_client import TMLLClient
from tmll.ml.modules.common.data_fetch import DataFetcher
from tmll.ml.modules.common.data_preprocess import DataPreprocessor
from tmll.ml.utils.document_generator import DocumentGenerator


class MemoryLeakSeverity(Enum):
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class MemoryMetrics:
    unreleased_allocations: int
    total_allocations: int
    leak_rate: float
    avg_allocation_size: float
    max_continuous_growth_duration: float
    memory_fragmentation_score: float
    regression_slope: float
    regression_intercept: float


@dataclass
class LeakAnalysisResult:
    severity: MemoryLeakSeverity
    confidence_score: float
    metrics: MemoryMetrics
    detected_patterns: List[str]
    suspicious_locations: pd.DataFrame


class MemoryLeakDetection(BaseModule):
    """
    Memory Leak Detection Module

    In this module, we analyze memory usage patterns to detect memory leaks in the data.
    Memory leaks occur when a program allocates memory but fails to release it, leading to memory exhaustion.
    The analysis consists of the following steps:
        1. Track pointer lifecycles to identify memory leaks
        2. Analyze memory usage trends to detect systematic growth
        3. Analyze allocation patterns to identify potential issues
        4. Calculate comprehensive memory metrics
        5. Evaluate the severity of memory issues and confidence in the assessment
        6. Identify suspicious memory allocation locations
        7. Collect and describe detected memory leak patterns

    Based on the analysis results, we can determine the severity of memory issues and provide insights into potential causes.
    """

    BASE_OUTPUTS = [
        Output.from_dict({
            'name': 'Events Table',
            'id': 'org.eclipse.tracecompass.internal.provisional.tmf.core.model.events.TmfEventTableDataProvider'
        }),
        Output.from_dict({
            'name': 'Memory Usage',
            'id': 'org.eclipse.tracecompass.lttng2.ust.core.analysis.memory.UstMemoryUsageDataProvider'
        }),
    ]

    def __init__(self, client: TMLLClient, experiment: Experiment):
        super().__init__(client, experiment)
        self.data_fetcher = DataFetcher(client)
        self.data_preprocessor = DataPreprocessor()
        self.dataframes = {}
        self.ptr_lifecycle = pd.DataFrame()

        self.slope_threshold = 0.5
        self.fragmentation_threshold = 0.7
        self.window_size = '1s'

    def process(self) -> None:
        """
        Process and prepare the data for memory leak analysis.
        """
        self.dataframes.clear()
        self.ptr_lifecycle = pd.DataFrame()

        # TODO: Have a flexible table line column ID configuration
        data, outputs = self.data_fetcher.fetch_data(
            experiment=self.experiment,
            target_outputs=self.BASE_OUTPUTS,
            table_line_column_ids=[4, 5, 17] # Column IDs for the Events Table (Event type, Contents, Timestamp ns)
        )

        if data is None:
            self.logger.error("No data fetched")
            return

        for output_key, output_data in data.items():
            shortened = output_key.split("$")[0]
            converted = next(iter(output for output in outputs if output.id == shortened), None) if outputs else None
            shortened = converted.name if converted else shortened

            if shortened not in self.dataframes:
                df = self.data_preprocessor.convert_to_datetime(output_data)
                if shortened == 'Events Table':
                    df = self._separate_events(df)

                if not df.empty:
                    self.dataframes[shortened] = df

    def _separate_events(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Separate memory allocation and deallocation events from other events.
        Also, we only keep the events related to memory allocation and deallocation.

        :param dataframe: The original events table
        :type dataframe: pd.DataFrame
        :return: The processed events table
        :rtype: pd.DataFrame
        """
        dataframe['event_category'] = 'other'
        dataframe.loc[dataframe['Event type'].str.contains('malloc', na=False), 'event_category'] = 'allocation'
        dataframe.loc[dataframe['Event type'].str.contains('free', na=False), 'event_category'] = 'deallocation'
        dataframe = dataframe[dataframe['event_category'] != 'other']

        dataframe = dataframe.rename({'size': 'allocation_size'}, axis=1)
        dataframe['allocation_size'] = dataframe['allocation_size'].astype(float)

        dataframe['ptr'] = dataframe['ptr'].astype(str)

        return dataframe

    def analyze_memory_leaks(self, window_size: str = '1s', fragmentation_threshold: float = 0.7,
                             slope_threshold: float = 0.5) -> LeakAnalysisResult:
        """
        Analyze memory usage patterns and detect memory leaks in the data.
        Here, the analysis consists of the following steps:
            1. Track pointer lifecycles to identify memory leaks
            2. Analyze memory usage trends to detect systematic growth
            3. Analyze allocation patterns to identify potential issues
            4. Calculate comprehensive memory metrics
            5. Evaluate the severity of memory issues and confidence in the assessment
            6. Identify suspicious memory allocation locations
            7. Collect and describe detected memory leak patterns

        :param window_size: The window size for trend analysis, defaults to '1s'
        :type window_size: str, optional
        :param fragmentation_threshold: The threshold for memory fragmentation, defaults to 0.7
        :type fragmentation_threshold: float, optional
        :param slope_threshold: The threshold for memory growth slope, defaults to 0.5
        :type slope_threshold: float, optional
        :return: The results of the memory leak analysis
        :rtype: LeakAnalysisResult
        """
        self.window_size = window_size
        self.fragmentation_threshold = fragmentation_threshold
        self.slope_threshold = slope_threshold

        ptr_tracking = self._track_pointer_lifecycle()
        memory_trend = self._analyze_memory_trend()
        allocation_patterns = self._analyze_allocation_patterns()
        metrics = self._calculate_memory_metrics(ptr_tracking, memory_trend, allocation_patterns)
        severity, confidence = self._evaluate_severity(metrics, memory_trend['p_value'])
        suspicious_locs = self._identify_suspicious_locations(ptr_tracking)
        patterns = self._collect_detected_patterns(memory_trend, allocation_patterns)

        return LeakAnalysisResult(
            severity=severity,
            confidence_score=confidence,
            metrics=metrics,
            detected_patterns=patterns,
            suspicious_locations=suspicious_locs
        )

    def _track_pointer_lifecycle(self) -> pd.DataFrame:
        """
        Track the lifecycle of memory pointers to identify memory leaks.

        :return: A DataFrame containing information about memory leaks
        :rtype: pd.DataFrame
        """
        if 'Events Table' not in self.dataframes or self.dataframes['Events Table'].empty:
            self.logger.warning("No events data available for memory leak analysis")
            return pd.DataFrame()

        if not self.ptr_lifecycle.empty:
            return self.ptr_lifecycle

        events_df = self.dataframes['Events Table']
        ptr_status = {}
        leak_records = []

        for idx, row in events_df.iterrows():
            ptr = row['ptr']
            event_type = row['event_category']

            if event_type == 'allocation':
                if ptr not in ptr_status:
                    ptr_status[ptr] = {
                        'allocation_time': idx,
                        'allocation_size': row['allocation_size']
                    }
            elif event_type == 'deallocation':
                if ptr in ptr_status:
                    lifetime = (idx - ptr_status[ptr]['allocation_time']).total_seconds()
                    leak_records.append({
                        'ptr': ptr,
                        'allocation_time': ptr_status[ptr]['allocation_time'],
                        'deallocation_time': idx,
                        'lifetime': lifetime,
                        'size': ptr_status[ptr]['allocation_size']
                    })
                    del ptr_status[ptr]

        # Add remaining unfreed pointers
        for ptr, info in ptr_status.items():
            leak_records.append({
                'ptr': ptr,
                'allocation_time': info['allocation_time'],
                'deallocation_time': None,
                'lifetime': None,
                'size': info['allocation_size']
            })

        self.ptr_lifecycle = pd.DataFrame(leak_records)

        return self.ptr_lifecycle

    def _analyze_memory_trend(self) -> Dict[str, Any]:
        """
        Analyze memory usage trends to detect systematic growth.

        :return: The results of the memory usage trend analysis
        :rtype: Dict[str, Any]
        """
        if 'Memory Usage' not in self.dataframes or self.dataframes['Memory Usage'].empty:
            self.logger.warning("No memory usage data available for trend analysis")
            return {}

        memory_df = self.dataframes['Memory Usage']

        # Calculate rolling statistics
        window_size = pd.Timedelta(self.window_size)
        rolling_mean = memory_df['Memory Usage'].rolling(window=window_size).mean()
        rolling_std = memory_df['Memory Usage'].rolling(window=window_size).std()

        # Perform linear regression
        slope, intercept, r_value, p_value, _ = stats.linregress(np.arange(len(memory_df)), memory_df['Memory Usage'].values)
        slope = cast(float, slope)
        intercept = cast(float, intercept)
        r_value = cast(float, r_value)
        p_value = cast(float, p_value)

        # Calculate growth characteristics
        is_significant = p_value < 0.05
        is_increasing = slope > 0
        growth_rate = slope if is_increasing else 0

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'growth_rate': growth_rate,
            'is_significant': is_significant,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std
        }

    def _analyze_allocation_patterns(self) -> Dict[str, Any]:
        """
        Analyze allocation patterns to identify potential issues.

        :return: The results of the allocation pattern analysis
        :rtype: Dict[str, Any]
        """
        if 'Events Table' not in self.dataframes or self.dataframes['Events Table'].empty:
            self.logger.warning("No events data available for allocation pattern analysis")
            return {}

        events_df = self.dataframes['Events Table']
        allocation_events = events_df[events_df['event_category'] == 'allocation']

        # Calculate allocation frequencies
        allocation_freq = allocation_events.resample('1min').size()

        # Analyze allocation sizes
        allocation_sizes = allocation_events['allocation_size'].dropna()

        return {
            'allocation_frequency': allocation_freq,
            'mean_allocation_size': allocation_sizes.mean(),
            'median_allocation_size': allocation_sizes.median(),
            'size_std': allocation_sizes.std(),
            'total_allocations': len(allocation_events),
            'unique_sizes': allocation_sizes.nunique()
        }

    def _calculate_memory_metrics(self, ptr_tracking: pd.DataFrame, memory_trend: Dict, allocation_patterns: Dict) -> MemoryMetrics:
        """
        Calculate comprehensive memory metrics to assess memory issues.
        Here, the metrics include:
            - Unreleased Allocations
            - Total Allocations
            - Leak Rate
            - Average Allocation Size
            - Maximum Continuous Growth Duration
            - Memory Fragmentation Score

        :param ptr_tracking: The DataFrame containing memory leak information
        :type ptr_tracking: pd.DataFrame
        :param memory_trend: The results of the memory usage trend analysis
        :type memory_trend: Dict
        :param allocation_patterns: The results of the allocation pattern analysis
        :type allocation_patterns: Dict
        :return: The calculated memory metrics
        :rtype: MemoryMetrics
        """
        unreleased = len(ptr_tracking[ptr_tracking['deallocation_time'].isna()])

        # Calculate memory fragmentation score
        if not ptr_tracking.empty:
            allocated_chunks = len(ptr_tracking)
            concurrent_chunks = len(ptr_tracking[ptr_tracking['deallocation_time'].isna()])
            fragmentation_score = concurrent_chunks / allocated_chunks if allocated_chunks > 0 else 0
        else:
            fragmentation_score = 0

        # Calculate maximum continuous growth duration
        rolling_mean = memory_trend['rolling_mean']
        growth_periods = (rolling_mean.diff() > 0).astype(int)
        max_growth_duration = (
            growth_periods.groupby((growth_periods != growth_periods.shift()).cumsum())
            .sum().max() * pd.Timedelta('1min').total_seconds()
        )

        return MemoryMetrics(
            unreleased_allocations=unreleased,
            total_allocations=allocation_patterns['total_allocations'],
            leak_rate=memory_trend['growth_rate'],
            avg_allocation_size=allocation_patterns['mean_allocation_size'],
            max_continuous_growth_duration=max_growth_duration,
            memory_fragmentation_score=fragmentation_score,
            regression_slope=memory_trend['slope'],
            regression_intercept=memory_trend['intercept']
        )

    def _evaluate_severity(self, metrics: MemoryMetrics, p_value: float) -> Tuple[MemoryLeakSeverity, float]:
        """
        Evaluate the severity of memory issues and confidence in the assessment.
        Severity indicates the impact of memory issues, while confidence reflects the reliability of the analysis.
        The severity levels are defined as follows:
            - NONE: No memory issues detected
            - LOW: Low impact memory issues detected
            - MEDIUM: Medium impact memory issues detected
            - HIGH: High impact memory issues detected
            - CRITICAL: Critical impact memory issues detected

        :param metrics: The calculated memory metrics
        :type metrics: MemoryMetrics
        :param p_value: The p-value of the memory usage trend analysis
        :type p_value: float
        :return: The severity of memory issues and confidence in the assessment
        :rtype: Tuple[MemoryLeakSeverity, float]
        """
        # Calculate base scores
        slop_threshold = 0.5
        fragmantation_threshold = 0.7
        growth_score = min(1.0, metrics.leak_rate / slop_threshold)
        unreleased_score = min(1.0, metrics.unreleased_allocations / metrics.total_allocations)
        fragmentation_score = min(1.0, metrics.memory_fragmentation_score / fragmantation_threshold)

        # Weight the scores
        weighted_score = (
            0.4 * growth_score +
            0.4 * unreleased_score +
            0.2 * fragmentation_score
        )

        # Calculate confidence based on data quality
        confidence = min(1.0, (
            0.6 * (1 - p_value) +
            0.4 * (metrics.total_allocations)
        ))

        # Determine severity
        if weighted_score < 0.2:
            severity = MemoryLeakSeverity.NONE
        elif weighted_score < 0.4:
            severity = MemoryLeakSeverity.LOW
        elif weighted_score < 0.6:
            severity = MemoryLeakSeverity.MEDIUM
        elif weighted_score < 0.8:
            severity = MemoryLeakSeverity.HIGH
        else:
            severity = MemoryLeakSeverity.CRITICAL

        return severity, confidence

    def _identify_suspicious_locations(self, ptr_tracking: pd.DataFrame) -> pd.DataFrame:
        """
        Identify suspicious memory allocation locations.
        These locations are characterized by large amounts of unreleased memory.

        :param ptr_tracking: The DataFrame containing memory leak information
        :type ptr_tracking: pd.DataFrame
        :return: The top suspicious memory allocation locations
        :rtype: pd.DataFrame
        """
        if 'Events Table' not in self.dataframes or self.dataframes['Events Table'].empty:
            self.logger.warning("No events data available for suspicious location analysis")
            return pd.DataFrame()

        events_df = self.dataframes['Events Table']

        # Find allocations without matching deallocations
        unfreed_ptrs = ptr_tracking[ptr_tracking['deallocation_time'].isna()]['ptr']

        # Get allocation events for unfreed pointers
        suspicious_allocs = events_df[
            (events_df['event_category'] == 'allocation') &
            (events_df['ptr'].isin(unfreed_ptrs))
        ]

        # Group by location and calculate metrics
        location_metrics = suspicious_allocs.groupby(['ptr']).agg({
            'allocation_size': ['sum', 'count'],
            'Event type': 'first'  # Keep the original event for context
        }).reset_index()

        location_metrics.columns = ['ptr', 'total_bytes', 'allocation_count', 'event_context']

        return location_metrics.sort_values('total_bytes', ascending=False)

    def _collect_detected_patterns(self, memory_trend: Dict[str, Any], allocation_patterns: Dict[str, Any]) -> List[str]:
        """
        Collect and describe detected memory leak patterns.
        These patterns provide insights into the detected memory issues. For example:
            - Systematic memory growth detected
            - Irregular allocation pattern detected
            - High memory usage volatility detected

        :param memory_trend: The results of the memory usage trend analysis
        :type memory_trend: Dict[str, Any]
        :param allocation_patterns: The results of the allocation pattern analysis
        :type allocation_patterns: Dict[str, Any]
        :return: The detected memory leak patterns
        :rtype: List[str]
        """
        patterns = []

        if memory_trend['is_significant'] and memory_trend['slope'] > 0:
            patterns.append(
                f"Systematic memory growth detected: {memory_trend['growth_rate']:.2f} bytes/second"
            )

        if allocation_patterns['allocation_frequency'].std() > allocation_patterns['allocation_frequency'].mean():
            patterns.append("Irregular allocation pattern detected")

        if memory_trend['rolling_std'].mean() > memory_trend['rolling_mean'].mean() * 0.1:
            patterns.append("High memory usage volatility detected")

        return patterns

    def _convert_bytes(self, size_in_bytes: float) -> str:
        """Convert bytes to a human-readable string with appropriate units."""
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = size_in_bytes
        unit_index = 0

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        return f"{size:.2f} {units[unit_index]}"

    def interpret(self, analysis_result: LeakAnalysisResult) -> None:
        """Interpret and display memory leak analysis results using the DocumentGenerator."""

        DocumentGenerator.section("Memory Leak Analysis Results")

        DocumentGenerator.metrics_group("Analysis Overview", {
            "Severity": analysis_result.severity.name,
            "Confidence Score": f"{analysis_result.confidence_score:.2f}"
        })

        DocumentGenerator.metrics_group("Memory Metrics", {
            "Unreleased Allocations": analysis_result.metrics.unreleased_allocations,
            "Total Allocations": analysis_result.metrics.total_allocations,
            "Leak Rate": f"{analysis_result.metrics.leak_rate:.2f} bytes/second",
            "Average Allocation Size": f"{analysis_result.metrics.avg_allocation_size:.2f} bytes",
            "Max Continuous Growth": f"{analysis_result.metrics.max_continuous_growth_duration:.2f} seconds",
            "Memory Fragmentation": f"{analysis_result.metrics.memory_fragmentation_score:.2f}"
        })

        DocumentGenerator.metrics_group("Detected Patterns", {
            f"{i + 1}": pattern for i, pattern in enumerate(analysis_result.detected_patterns)
        })

        if not analysis_result.suspicious_locations.empty:
            suspicious_headers = ["Pointer", "Total Bytes", "Allocation Count", "Event Context"]
            suspicious_rows = [
                [row['ptr'], row['total_bytes'], row['allocation_count'], row['event_context']]
                for _, row in analysis_result.suspicious_locations.head().iterrows()
            ]
            DocumentGenerator.table(
                suspicious_headers,
                suspicious_rows,
                "Top 5 Suspicious Locations"
            )

        memory_df = self.dataframes['Memory Usage']
        DocumentGenerator.metrics_group("Memory Usage Statistics", {
            "Peak Memory Usage": self._convert_bytes(memory_df['Memory Usage'].max()),
            "Average Memory Usage": self._convert_bytes(memory_df['Memory Usage'].mean()),
            "Memory Usage Std Dev": self._convert_bytes(memory_df['Memory Usage'].std())
        })

        allocation_events = self.dataframes['Events Table'][
            self.dataframes['Events Table']['event_category'] == 'allocation'
        ]
        deallocation_events = self.dataframes['Events Table'][
            self.dataframes['Events Table']['event_category'] == 'deallocation'
        ]

        DocumentGenerator.metrics_group("Allocation Statistics", {
            "Total Allocations": f"{len(allocation_events):,}",
            "Total Deallocations": f"{len(deallocation_events):,}",
            "Unmatched Allocations": f"{len(allocation_events) - len(deallocation_events):,}"
        })

        ptr_tracking = self._track_pointer_lifecycle()
        lifetimes = ptr_tracking['lifetime'].dropna()
        if not lifetimes.empty:
            DocumentGenerator.metrics_group("Pointer Lifetime Statistics", {
                "Average Lifetime": f"{lifetimes.mean():.2f} seconds",
                "Median Lifetime": f"{lifetimes.median():.2f} seconds",
                "Maximum Lifetime": f"{lifetimes.max():.2f} seconds"
            })

    def plot(self, analysis_result: LeakAnalysisResult) -> None:
        """
        Plot memory usage trends and analysis results.

        :param analysis_result: The results of the memory leak analysis
        :type analysis_result: LeakAnalysisResult
        """
        memory_df = self.dataframes['Memory Usage']
        events_df = self.dataframes['Events Table']
        allocation_events = events_df[events_df['event_category'] == 'allocation']
        deallocation_events = events_df[events_df['event_category'] == 'deallocation']
        ptr_tracking = self._track_pointer_lifecycle()
        lifetimes = ptr_tracking['lifetime'].dropna()

        # Perform linear regression on memory usage
        slope = analysis_result.metrics.regression_slope
        intercept = analysis_result.metrics.regression_intercept

        # Use a standard color palette
        colors = plt.get_cmap('tab10')

        # Plot 1: Memory Usage Over Time
        plots = [
            {
                "plot_type": "time_series",
                "data": memory_df,
                "y": "Memory Usage",
                "label": "Memory Usage",
                "alpha": 0.8,
                "linewidth": 2.5,
                "color": colors(0)
            },
            {
                "plot_type": "time_series",
                "data": memory_df.rolling(window=self.window_size).mean(),
                "y": "Memory Usage",
                "label": "Rolling Mean",
                "alpha": 0.9,
                "linewidth": 2.5,
                "color": colors(1)
            },
            {
                "plot_type": "time_series",
                "data": pd.DataFrame({"timestamp": memory_df.index, "trend_line": slope * np.arange(len(memory_df)) + intercept}),
                "x": "timestamp",
                "y": "trend_line",
                "label": f"Trend (slope: {slope:.2f})",
                "color": colors(2),
                "linestyle": "--",
                "alpha": 0.8,
                "linewidth": 2.5
            }
        ]
        self._plot(plots, plot_size=(15, 5), dpi=300, fig_title="Memory Usage Over Time",
                   fig_xlabel="Time", fig_ylabel="Memory Usage (bytes)", grid=True)

        # Plot 2: Allocation Patterns
        plots = [
            {
                "plot_type": "time_series",
                "data": allocation_events.resample(self.window_size).size().to_frame(name='Allocations'),
                "y": "Allocations",
                "label": "Allocations",
                "alpha": 0.8,
                "color": colors(3),
                "linewidth": 2.5
            },
            {
                "plot_type": "time_series",
                "data": deallocation_events.resample(self.window_size).size().to_frame(name='Deallocations'),
                "y": "Deallocations",
                "label": "Deallocations",
                "alpha": 0.8,
                "color": colors(4),
                "linewidth": 2.5
            }
        ]
        self._plot(plots, plot_size=(15, 5), dpi=300, fig_title="Memory Operations Over Time",
                   fig_xlabel="Time", fig_ylabel="Operations per Second", grid=True)

        # Plot 3: Pointer Lifetime Distribution
        plots = [
            {
                "plot_type": "histogram",
                "data": lifetimes,
                "bins": 50,
                "alpha": 0.8,
                "color": colors(5)
            },
            {
                "plot_type": "vline",
                "data": None,
                "x": lifetimes.mean(),
                "label": f"Mean: {lifetimes.mean():.2f}s",
                "color": colors(6),
                "linestyle": "--",
                "linewidth": 2.5
            },
            {
                "plot_type": "vline",
                "data": None,
                "x": lifetimes.median(),
                "label": f"Median: {lifetimes.median():.2f}s",
                "color": colors(7),
                "linestyle": "--",
                "linewidth": 2.5
            }
        ]
        self._plot(plots, plot_size=(15, 5), dpi=300, fig_title="Pointer Lifetime Distribution",
                   fig_xlabel="Lifetime (seconds)", fig_ylabel="Count", grid=True)

        # Plot 4: Memory Fragmentation Analysis
        alloc_cumsum = allocation_events.resample(self.window_size).size().cumsum()
        dealloc_cumsum = deallocation_events.resample(self.window_size).size().cumsum()
        active_allocations = alloc_cumsum - dealloc_cumsum
        total_ops = alloc_cumsum + dealloc_cumsum
        fragmentation_score = (active_allocations / total_ops.replace(0, 1)) * 100

        plots = [
            {
                "plot_type": "time_series",
                "data": fragmentation_score.to_frame(name='Fragmentation Score'),
                "y": "Fragmentation Score",
                "label": "Fragmentation Score",
                "color": colors(8),
                "alpha": 0.8,
                "linewidth": 2.5
            },
            {
                "plot_type": "hline",
                "data": None,
                "y": self.fragmentation_threshold * 100,
                "label": f"Threshold ({self.fragmentation_threshold * 100}%)",
                "color": colors(9),
                "linestyle": "--",
                "linewidth": 2.5
            }
        ]
        self._plot(plots, plot_size=(15, 5), dpi=300, fig_title="Memory Fragmentation Analysis",
                   fig_xlabel="Time", fig_ylabel="Fragmentation Score (%)", grid=True)
