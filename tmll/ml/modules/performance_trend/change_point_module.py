import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from itertools import product

from tmll.ml.modules.base_module import BaseModule
from tmll.common.models.experiment import Experiment
from tmll.common.models.output import Output
from tmll.tmll_client import TMLLClient
from tmll.ml.modules.common.data_fetch import DataFetcher
from tmll.ml.modules.common.data_preprocess import DataPreprocessor


class ChangePointAnalysis(BaseModule):
    """
    Change point detection module for analyzing performance trends.

    Basically, this module aims to indicate the significant changes that happens
    in the performance metrics of a system (e.g., CPU usage, memory usage, etc.).

    The module supports multiple methods for change point detection:
    - Single metric analysis: Detects change points in individual metrics (e.g., CPU usage)
    - Z-score analysis: Combines metrics using z-score normalization and detects change points
    - Voting-based analysis: Combines change points from individual metrics using voting
    - PCA analysis: Uses principal component analysis to detect change points
    """

    AVAILABLE_MODELS = {
        'rbf': ruptures.Binseg(model="rbf"),
        'linear': ruptures.Binseg(model="linear"),
        'normal': ruptures.Binseg(model="normal"),
        'cosine': ruptures.Binseg(model="cosine")
    }

    AVAILABLE_METRICS = ['single', 'zscore', 'voting', 'pca']

    def __init__(self, client: TMLLClient, experiment: Experiment,
                 min_size: int = 10, window_size: int = 5):
        super().__init__(client, experiment)

        self.min_size = min_size
        self.window_size = window_size
        self.data_fetcher = DataFetcher(client)
        self.data_preprocessor = DataPreprocessor()
        self.dataframes = {}
        self.timestamps = None
        self.combined_df = None

        self.best_params = {}

    def process(self, outputs: Optional[List[Output]] = None):
        """
        Fetch and process data for change point detection.

        :param outputs: List of outputs to fetch data for
        :type outputs: Optional[List[Output]]
        """
        self.dataframes.clear()

        data = self.data_fetcher.fetch_data(
            experiment=self.experiment,
            target_outputs=outputs
        )

        if data is None:
            self.logger.error("No data fetched")
            return

        for output_key, output_data in data.items():
            shortened = output_key.split("$")[0]
            converted = next(iter(
                output for output in outputs if output.id == shortened), None) if outputs else None
            shortened = converted.name if converted else shortened

            if shortened not in self.dataframes:
                df = self.data_preprocessor.normalize(output_data)
                df = self.data_preprocessor.convert_to_datetime(df)
                df = self.data_preprocessor.remove_minimum(df)
                df = self.data_preprocessor.resample(df, frequency='1ms')
                self.dataframes[shortened] = df

        self.dataframes = {
            name: df for name, df in self.dataframes.items()
            if len(df) > 1
        }

        self.dataframes, self.timestamps = DataPreprocessor.align_timestamps(
            self.dataframes)

        self.combined_df = self.data_preprocessor.combine_dataframes(
            list(self.dataframes.values()))

    def _calculate_changes_magnitude(self, data: np.ndarray, change_points: List[int]) -> List[float]:
        """
        Calculate magnitude of changes at change points

        :param data: Input time series data
        :type data: np.ndarray
        :param change_points: Detected change points
        :type change_points: List[int]
        :return: Magnitudes of changes
        :rtype: List[float]
        """
        magnitudes = []

        for cp in change_points:
            # Calculate window size based on data
            window_size = min(self.window_size, cp, len(data) - cp)

            # Get segments before and after change point
            before_segment = data[max(0, cp - window_size):cp]
            after_segment = data[cp:min(len(data), cp + window_size)]

            if len(before_segment) == 0 or len(after_segment) == 0:
                magnitudes.append(0)
                continue

            # Calculate statistical properties
            before_mean = np.mean(before_segment)
            after_mean = np.mean(after_segment)
            before_std = np.std(before_segment)
            after_std = np.std(after_segment)

            # Calculate magnitude components
            mean_change = abs(after_mean - before_mean)
            var_change = abs(after_std - before_std)
            snr = mean_change / \
                max(before_std, after_std) if max(
                    before_std, after_std) > 0 else mean_change

            # Combined magnitude score
            magnitude = (0.5 * snr) + (0.3 * mean_change) + (0.2 * var_change)
            magnitudes.append(magnitude)

        return magnitudes

    def _calculate_statistical_metrics(self, segment_before: np.ndarray, segment_after: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate statistical metrics for segments before and after a change point.
        The statistical metrics are: mean change, variance change, and signal-to-noise ratio.

        :param segment_before: Data segment before change point
        :type segment_before: np.ndarray
        :param segment_after: Data segment after change point
        :type segment_after: np.ndarray
        :return: Tuple of (mean_change, var_change, snr)
        :rtype: Tuple[float, float, float]
        """
        if len(segment_before) == 0 or len(segment_after) == 0:
            return 0.0, 0.0, 0.0

        before_mean = np.mean(segment_before)
        after_mean = np.mean(segment_after)
        before_std = np.std(segment_before)
        after_std = np.std(segment_after)

        mean_change = abs(after_mean - before_mean)
        var_change = abs(after_std - before_std)
        noise = max(before_std, after_std)
        snr = mean_change / noise if noise > 0 else mean_change

        return float(mean_change), float(var_change), float(snr)

    def _get_segments(self, data: np.ndarray, change_point: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get segments before and after a change point.

        :param data: Input time series data
        :type data: np.ndarray
        :param change_point: Change point index
        :type change_point: int
        :return: Tuple of (before_segment, after_segment)
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        window_size = min(self.window_size, change_point,
                          len(data) - change_point)
        before_segment = data[max(0, change_point - window_size):change_point]
        after_segment = data[change_point:min(
            len(data), change_point + window_size)]
        return before_segment, after_segment

    def _calculate_significance(self, mean_change: float, var_change: float, snr: float) -> float:
        """
        Calculate significance score from statistical metrics.

        :param mean_change: Change in mean
        :type mean_change: float
        :param var_change: Change in variance
        :type var_change: float
        :param snr: Signal-to-noise ratio
        :type snr: float
        :return: Combined significance score
        :rtype: float
        """
        return (0.5 * snr) + (0.3 * mean_change) + (0.2 * var_change)

    def _detect_changes(self, data: np.ndarray, n_change_points: int = 5, tune_hyperparameters: bool = True, method_key: Optional[str] = None) -> List[int]:
        """
        Detect the most significant change points in order of their importance.
        We want to detect the most significant change points in the data.

        :param data: Input time series data
        :type data: np.ndarray
        :param n_change_points: Number of change points to detect
        :type n_change_points: int
        :param method_key: Key for storing method-specific parameters
        :type method_key: Optional[str]
        :return: Detected change points
        :rtype: List[int]
        """
        # Tune parameters if needed
        if tune_hyperparameters and method_key and method_key not in self.best_params:
            self.best_params[method_key] = self._tune_hyperparameters(data)

        params = self.best_params.get(method_key, {}) if method_key else {}
        model_type = params.get('model', 'rbf')
        min_size = params.get('min_size', 5)
        jump = params.get('jump', 2)

        # Create and fit model
        model = ruptures.Binseg(model=model_type, min_size=min_size, jump=jump)
        model.fit(data.reshape(-1, 1))

        # Get extra change points for significance evaluation
        # We want to ensure that we have enough points to evaluate significance
        n_extra = min(len(data) // 2, n_change_points * 3)
        all_change_points = model.predict(n_bkps=n_extra)[:-1]

        if not all_change_points:
            return []

        # Calculate significance for each point
        significance_scores = []
        for cp in all_change_points:
            before_segment, after_segment = self._get_segments(data, cp)
            if len(before_segment) == 0 or len(after_segment) == 0:
                continue

            mean_change, var_change, snr = self._calculate_statistical_metrics(
                before_segment, after_segment)
            significance = self._calculate_significance(
                mean_change, var_change, snr)
            significance_scores.append((cp, significance))

        # Select top points by significance
        significant_points = sorted(
            significance_scores, key=lambda x: x[1], reverse=True)
        return sorted(point[0] for point in significant_points[:n_change_points])

    def get_change_points(self, metrics: Optional[List[str]] = None, n_change_points: int = 3,
                          methods: Optional[List[str]] = None,
                          tune_hyperparameters: bool = True) -> Dict:
        """
        Find the most significant change points in the data using multiple analysis methods.
        The methods include:
        - Single metric analysis: Detects change points in individual metrics (e.g., CPU usage)
        - Z-score analysis: Combines metrics using z-score normalization and detects change points (combined score)
        - Voting-based analysis: Combines change points from individual metrics using voting (majority voting)
        - PCA analysis: Uses principal component analysis to detect change points (principal component)

        :param metrics: List of metrics to analyze (None for all metrics)
        :type metrics: Optional[List[str]]
        :param n_change_points: Number of change points to detect
        :type n_change_points: int
        :param methods: Analysis methods to use (e.g., 'single', 'zscore', 'voting', 'pca')
        :type methods: Optional[List[str]]
        :param annotations: Optional annotations for change points
        :type annotations: Optional[List[str]]
        :return: Analysis results for each method
        :rtype: Dict
        """
        if self.combined_df is None:
            self.logger.error("No data available for analysis")
            return {}

        results = {}
        metrics_to_analyze = metrics if metrics else list(self.dataframes.keys())

        if not methods:
            methods = self.AVAILABLE_METRICS

        # Single metric analysis
        if 'single' in methods:
            results['single'] = {}
            for metric in metrics_to_analyze:
                if metric not in self.dataframes:
                    self.logger.warning(f"Metric {metric} not found")
                    continue

                data = self.dataframes[metric].iloc[:, 0].values
                change_points = self._detect_changes(data=np.array(data),
                                                     n_change_points=n_change_points,
                                                     tune_hyperparameters=tune_hyperparameters,
                                                     method_key=f'single_{metric}')
                magnitudes = self._calculate_changes_magnitude(np.array(data), change_points)

                results['single'][metric] = {
                    'change_points': change_points,
                    'magnitudes': magnitudes
                }

        # Z-score analysis
        if 'zscore' in methods:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(
                self.combined_df[metrics_to_analyze])
            combined_zscore = np.sqrt(np.mean(np.square(scaled_data), axis=1))

            change_points = self._detect_changes(data=combined_zscore,
                                                 n_change_points=n_change_points,
                                                 tune_hyperparameters=tune_hyperparameters,
                                                 method_key='zscore')
            magnitudes = self._calculate_changes_magnitude(combined_zscore, change_points)

            results['zscore'] = {
                'change_points': change_points,
                'magnitudes': magnitudes,
                'combined_score': combined_zscore
            }

        # Voting-based analysis
        if 'voting' in methods:
            vote_matrix = np.zeros(len(self.combined_df))
            individual_changes = {}

            for metric in metrics_to_analyze:
                data = self.combined_df[metric].values
                changes = self._detect_changes(data=np.array(data),
                                               n_change_points=n_change_points,
                                               tune_hyperparameters=tune_hyperparameters,
                                               method_key=f'voting_{metric}')
                individual_changes[metric] = changes

                for cp in changes:
                    window = slice(max(0, cp - self.window_size), min(len(vote_matrix), cp + self.window_size + 1))
                    vote_matrix[window] += 1

            change_points = self._detect_changes(data=vote_matrix,
                                                 n_change_points=n_change_points,
                                                 tune_hyperparameters=tune_hyperparameters,
                                                 method_key='voting')
            magnitudes = self._calculate_changes_magnitude(vote_matrix, change_points)

            results['voting'] = {
                'change_points': change_points,
                'magnitudes': magnitudes,
                'vote_matrix': vote_matrix,
                'individual_changes': individual_changes
            }

        # PCA analysis
        if 'pca' in methods:
            pca = PCA(n_components=1)
            principal_component = pca.fit_transform(self.combined_df[metrics_to_analyze]).flatten()

            change_points = self._detect_changes(data=principal_component,
                                                 n_change_points=n_change_points,
                                                 tune_hyperparameters=tune_hyperparameters,
                                                 method_key='pca')
            magnitudes = self._calculate_changes_magnitude(principal_component, change_points)

            results['pca'] = {
                'change_points': change_points,
                'magnitudes': magnitudes,
                'principal_component': principal_component,
                'explained_variance_ratio': pca.explained_variance_ratio_[0]
            }

        return results

    def _tune_hyperparameters(self, data: np.ndarray, param_grid: Optional[Dict] = None,
                              cv_splits: int = 5, deep_search: bool = False) -> Dict:
        """
        Tune hyperparameters for change point detection using cross-validation.

        :param data: Input time series data
        :type data: np.ndarray
        :param param_grid: Parameter grid for hyperparameter tuning
        :type param_grid: Optional[Dict]
        :param cv_splits: Number of cross-validation splits
        :type cv_splits: int
        :param deep_search: Perform a deeper search for hyperparameters
        :type deep_search: bool
        :return: Best hyperparameters
        :rtype: Dict
        """
        if param_grid is None:
            param_grid = {
                'model': ['rbf', 'linear', 'normal', 'cosine'],  # Model type
                'min_size': [5, 10, 20],  # Minimum size of a segment
                'window_size': [3, 5],  # Window size for calculating change magnitude
                'penalty': [1, 3],  # Penalty parameter for change point detection
                'jump': [1, 3]  # Jump parameter for computational efficiency
            }

        if deep_search:
            param_grid['min_size'] = [5, 10, 15, 20, 25, 30, 35, 40]
            param_grid['window_size'] = [3, 5, 7, 10, 15]
            param_grid['penalty'] = [1, 2, 3, 4, 5]
            param_grid['jump'] = [1, 2, 3, 4, 5]

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        best_score = float('inf')
        best_params = {}

        # Generate all parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

        for params in param_combinations:
            scores = []

            for train_idx, test_idx in tscv.split(data):
                train_data = data[train_idx]
                test_data = data[test_idx]

                model = ruptures.Binseg(
                    model=params['model'],
                    min_size=params['min_size'],
                    jump=params['jump']
                )

                try:
                    model.fit(train_data.reshape(-1, 1))

                    bkps = model.predict(pen=params['penalty'])

                    error = self._calculate_prediction_error(test_data, bkps)
                    scores.append(error)
                except Exception as e:
                    self.logger.warning(f"Error with parameters {
                                        params}: {str(e)}")
                    scores.append(float('inf'))

            mean_score = np.mean(scores)

            if mean_score < best_score:
                best_score = mean_score
                best_params = params

        self.best_params = best_params
        return best_params

    def _calculate_prediction_error(self, data: np.ndarray, change_points: List[int]) -> float:
        """
        Calculate prediction error for a given set of change points.
        Uses mean squared error between actual values and piecewise constant approximation.

        :param data: Input time series data
        :type data: np.ndarray
        :param change_points: Detected change points
        :type change_points: List[int]
        :return: Prediction error
        :rtype: float
        """
        if len(change_points) < 2:
            return float('inf')

        error = 0
        start_idx = 0

        for end_idx in change_points[:-1]:
            segment = data[start_idx:end_idx]
            segment_mean = np.mean(segment)
            error += np.sum((segment - segment_mean) ** 2)
            start_idx = end_idx

        return float(error / len(data))

    def plot(self, results: Dict) -> None:
        """
        Plot change points and analysis results.

        :param results: Results from `get_change_points` containing all analysis methods
        :type results: Dict
        """
        if self.combined_df is None:
            self.logger.error("Combined DataFrame is None")
            return

        combined_plots = []
        colors = plt.get_cmap('rainbow')
        for idx, metric in enumerate(self.combined_df.columns):
            # Add time series plot for each metric (combined)
            combined_plots.append({
                'plot_type': 'time_series',
                'data': pd.DataFrame({metric: self.combined_df[metric]}, index=self.combined_df.index),
                'label': metric,
                'title': metric,
                'y': metric,
                'color': colors(idx / len(self.combined_df.columns)),
                'linewidth': 1.5,
                'xlabel': 'Time',
                'ylabel': metric
            })
        self._plot(plots=combined_plots, plot_size=(15, 3), dpi=300,
                   fig_title="Time Series of Metrics",
                   fig_xlabel='Time',
                   fig_ylabel='Value (normalized)')

        # Plot individual metrics if 'single' method is in results
        if method := results.get('single'):
            for metric in self.combined_df.columns:
                plots = []

                # Add individual metric plot
                plots.append({
                    'plot_type': 'time_series',
                    'data': pd.DataFrame({metric: self.combined_df[metric]}, index=self.combined_df.index),
                    'title': metric,
                    'label': metric,
                    'y': metric,
                    'color': colors(0),
                    'linewidth': 1.5
                })

                # Add change points if single analysis was performed for this metric
                if metric in results['single']:
                    metric_result = results['single'][metric]
                    for cp, mag in zip(metric_result['change_points'], metric_result['magnitudes']):
                        change_time = self.combined_df.index[cp]

                        # Add change point to plot
                        plots.append({
                            'plot_type': 'vline',
                            'data': None,
                            'label': f"Change Point (Δ={mag:.2f})",
                            'x': change_time,
                            'color': 'red',
                            'linestyle': '--',
                            'alpha': 0.75,
                            'linewidth': 1.5
                        })
                        
                        # Add annotation to plot
                        plots.append({
                            'plot_type': 'annotate',
                            'data': None,
                            'xy':change_time,
                            'text': f'Δ={mag:.2f}',
                            'color': 'red'
                        })

                self._plot(plots=plots, plot_size=(15, 3), dpi=300,
                           fig_title=f"Change Points for {metric}",
                           fig_xlabel='Time',
                           fig_ylabel=f'{metric} (normalized)')

        # Plot aggregate analysis methods
        for method in ['zscore', 'voting', 'pca']:
            if method not in results:
                continue

            result = results[method]
            title = ""
            plots = []
            data = None

            if method == 'zscore':
                data = result['combined_score']
                title = "Combined Z-score Analysis"

            elif method == 'voting':
                data = result['vote_matrix']
                title = "Voting-based Analysis"

            elif method == 'pca':
                data = result['principal_component']
                title = f"PCA Analysis (Explained Variance: {
                    result['explained_variance_ratio']:.2f})"

            # Add combined plot
            plots.append({
                'plot_type': 'time_series',
                'data': pd.DataFrame({'Combined': data}, index=self.combined_df.index),
                'label': 'Combined',
                'title': title,
                'y': 'Combined',
                'color': colors(0),
                'linewidth': 1.5
            })

            # Add change points for aggregate methods
            for cp, mag in zip(result['change_points'], result['magnitudes']):
                change_time = self.combined_df.index[cp]
                plots.append({
                    'plot_type': 'vline',
                    'data': None,
                    'label': f"Change Point (Δ={mag:.2f})",
                    'x': change_time,
                    'color': 'red',
                    'linestyle': '--',
                    'alpha': 0.75,
                    'linewidth': 1.5
                })

                # Add annotation to plot
                plots.append({
                    'plot_type': 'annotate',
                    'data': None,
                    'xy':change_time,
                    'text': f'Δ={mag:.2f}',
                    'color': 'red'
                })

            self._plot(plots=plots, plot_size=(15, 3), dpi=300,
                       fig_title=title,
                       fig_xlabel='Time',
                       fig_ylabel='Combined Score')
