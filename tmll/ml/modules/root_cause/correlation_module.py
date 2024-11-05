import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kendalltau

from tmll.ml.modules.base_module import BaseModule
from tmll.common.models.experiment import Experiment
from tmll.common.models.output import Output
from tmll.tmll_client import TMLLClient
from tmll.ml.modules.common.data_fetch import DataFetcher
from tmll.ml.modules.common.data_preprocess import DataPreprocessor
from tmll.ml.modules.common.statistics import Statistics


class CorrelationAnalysis(BaseModule):
    """
    Correlation analysis module to analyze correlations between multiple time series data.
    Here, we calculate the correlation between all pairs of time series dataframes and
    analyze the correlations during a specific time period.
    """

    def __init__(self, client: TMLLClient, experiment: Experiment):
        """
        Initialize the correlation analysis module with the given TMLL client and experiment.
        
        :param client: The TMLL client to use
        :type client: TMLLClient
        :param experiment: The experiment to analyze
        :type experiment: Experiment
        """
        super().__init__(client, experiment)

        self.data_fetcher = DataFetcher(client)
        self.data_preprocessor = DataPreprocessor()
        self.dataframes = {}

    def process(self, outputs: Optional[List[Output]] = None):
        """
        Initialize the correlator with multiple named dataframes.

        :param outputs: List of outputs to fetch data for
        :type outputs: Optional[List[Output]]
        """
        self.dataframes.clear()

        data = self.data_fetcher.fetch_data(
            experiment=self.experiment, target_outputs=outputs)
        if data is None:
            self.logger.error("No data fetched")
            return

        for output_key, output_data in data.items():
            shortened = output_key.split("$")[0]
            converted = next(iter(
                output for output in outputs if output.id == shortened), None) if outputs else None
            shortened = converted.name if converted else shortened
            if shortened not in self.dataframes:
                self.dataframes[shortened] = self.data_preprocessor.normalize(
                    output_data)
                self.dataframes[shortened] = self.data_preprocessor.convert_to_datetime(
                    self.dataframes[shortened])
                self.dataframes[shortened] = self.data_preprocessor.resample(
                    self.dataframes[shortened], frequency='1ms')

        # Remove dataframes with data less than 2
        self.dataframes = {name: df for name,
                           df in self.dataframes.items() if len(df) > 1}

        # Align dataframes' timestamps
        self.dataframes, self.timestamps = DataPreprocessor.align_timestamps(
            self.dataframes)

    def _calculate_correlation(self, series1: pd.Series, series2: pd.Series,
                              method: Optional[str] = None) -> Tuple[float, float]:
        """Calculate correlation between two series using the specified or automatically determined method.
        
        :param series1: First series to compare
        :type series1: pd.Series
        :param series2: Second series to compare
        :type series2: pd.Series
        :param method: Correlation method to use (pearson, kendall, spearman), defaults to None
        :type method: Optional[str], optional
        :return: Correlation coefficient and p-value
        :rtype: Tuple[float, float]
        """
        if method is None:
            method = Statistics.get_correlation_method(series1, series2)

        if method == 'pearson':
            return pearsonr(series1, series2)
        elif method == 'spearman':
            return spearmanr(series1, series2)
        else:  # kendall
            return kendalltau(series1, series2)

    def analyze_period(self, start_time: Optional[pd.Timestamp] = None,
                       end_time: Optional[pd.Timestamp] = None,
                       method: Optional[str] = None) -> Dict:
        """
        Analyze correlations between all dataframes. Optionally, specify a time period and correlation method.

        :param start_time: Start time of the period to analyze
        :type start_time: Optional[pd.Timestamp], optional
        :param end_time: End time of the period to analyze
        :type end_time: Optional[pd.Timestamp], optional
        :param method: Correlation method to use (pearson, kendall, spearman), defaults to None
        :type method: Optional[str], optional
        :return: Correlation analysis results
        :rtype: Dict
        """
        results = {
            'period': {'start': start_time, 'end': end_time},
            'correlations': {},
            'methods_used': {}
        }

        # Filter data for the specified period
        if start_time and end_time:
            period_data = {
                name: df[(df.index >= start_time) & (df.index <= end_time)]
                for name, df in self.dataframes.items()
            }
        else:
            period_data = self.dataframes

        # Remove dataframes with no data in the period
        period_data = {name: df for name,
                       df in period_data.items() if not df.empty}

        # Calculate all pairwise correlations
        names = list(period_data.keys())
        correlation_matrix = pd.DataFrame(
            index=names, columns=names, dtype=float)
        p_values_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
        methods_matrix = pd.DataFrame(index=names, columns=names, dtype=str)

        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i < j:  # Only calculate upper triangle
                    series1 = period_data[name1][name1]
                    series2 = period_data[name2][name2]

                    corr, p_value = self._calculate_correlation(
                        series1, series2, method)
                    used_method = method or Statistics.get_correlation_method(
                        series1, series2)

                    correlation_matrix.loc[name1, name2] = corr
                    correlation_matrix.loc[name2, name1] = corr
                    p_values_matrix.loc[name1, name2] = p_value
                    p_values_matrix.loc[name2, name1] = p_value
                    methods_matrix.loc[name1, name2] = used_method
                    methods_matrix.loc[name2, name1] = used_method
                elif i == j:
                    correlation_matrix.loc[name1, name2] = 1.0
                    p_values_matrix.loc[name1, name2] = 0.0
                    methods_matrix.loc[name1, name2] = 'identity'

        results['correlations'] = correlation_matrix
        results['p_values'] = p_values_matrix
        results['methods'] = methods_matrix

        return results

    def analyze_lags(self, series1_name: str, series2_name: str,
                     max_lag: int = 10) -> Dict:
        """
        Perform lag correlation analysis between two specific time series.
        Basically, we shift one series by a lag and calculate the correlation with the other series.
        Accordingly, we may find the optimal lag that maximizes the correlation. This is useful for 
        cases where one series is leading or lagging the other.

        :param series1_name: Name of the first series to compare
        :type series1_name: str
        :param series2_name: Name of the second series to compare
        :type series2_name: str
        :param max_lag: Maximum lag to consider, defaults to 10
        :type max_lag: int, optional
        :return: Lag analysis results
        :rtype: Dict
        """
        series1 = self.dataframes[series1_name][series1_name]
        series2 = self.dataframes[series2_name][series2_name]

        lag_results = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                s1 = series1.iloc[abs(lag):]
                s2 = series2.iloc[:lag]
            elif lag > 0:
                s1 = series1.iloc[:-lag]
                s2 = series2.iloc[lag:]
            else:
                s1 = series1
                s2 = series2

            corr, p_value = self._calculate_correlation(s1, s2)
            lag_results.append({
                'lag': lag,
                'correlation': corr,
                'p_value': p_value
            })

        # Find optimal lag
        optimal_lag = max(lag_results, key=lambda x: abs(x['correlation']))

        return {
            'lag_analysis': lag_results,
            'optimal_lag': optimal_lag,
            'series1': series1_name,
            'series2': series2_name
        }

    def plot_correlation_matrix(self, results: Dict) -> None:
        """Plot correlation matrix heatmap from analysis results.
        
        :param results: Correlation analysis results
        :type results: Dict
        """
        # Create mask for upper triangle
        mask = np.zeros_like(results['correlations'])
        mask[np.triu_indices_from(mask)] = True

        self._plot([{
            "plot_type": "heatmap",
            "data": results['correlations'].fillna(0),
            "mask": mask,
            "cmap": "RdBu",
        }], plot_size=(10, 8), fig_title="Correlation Matrix")

    def plot_lag_analysis(self, lag_results: Dict) -> None:
        """Plot lag correlation analysis results.
        
        :param lag_results: Lag analysis results
        :type lag_results: Dict
        """
        lags = [x['lag'] for x in lag_results['lag_analysis']]
        correlations = [x['correlation'] for x in lag_results['lag_analysis']]
        optimal_lag = lag_results['optimal_lag']['lag']
        optimal_corr = lag_results['optimal_lag']['correlation']

        data = pd.DataFrame({'lag': lags, 'correlation': correlations})
        optimal_point = pd.DataFrame({'lag': [optimal_lag], 'correlation': [optimal_corr]})

        plots = [
            {
                "plot_type": "time_series",
                "data": data,
                "x": "lag",
                "y": "correlation",
                "marker": "o",
                "label": "Correlation",
                "alpha": 0.75,
                "color": "blue"
            },
            {
                "plot_type": "scatter",
                "data": optimal_point,
                "x": "lag",
                "y": "correlation",
                "label": f"Optimal Lag ({optimal_lag})\nCorrelation: {optimal_corr:.2f}",
                "color": "red",
                "is_top": True,
                "alpha": 0.8,
                "marker": "*",
                "s": 200
            },
            {
                "plot_type": "vline",
                "data": None,
                "x": 0,
                "color": "red",
                "linestyle": "--",
                "alpha": 0.3
            },
            {
                "plot_type": "hline",
                "data": None,
                "y": 0,
                "color": "red",
                "linestyle": "--",
                "alpha": 0.3
            }
        ]

        self._plot(plots, plot_size=(10, 6),
                   fig_title=f"Lag Analysis: {lag_results['series1']} vs {lag_results['series2']}",
                   fig_xlabel="Lag", fig_ylabel="Correlation")

    def plot_time_series(self, series: List[str],
                         start_time: Optional[pd.Timestamp] = None,
                         end_time: Optional[pd.Timestamp] = None) -> None:
        """Plot two time series together for visual comparison.
        
        :param series: List of series names to plot
        :type series: List[str]
        :param start_time: Start time of the period to plot
        :type start_time: Optional[pd.Timestamp]
        :param end_time: End time of the period to plot
        :type end_time: Optional[pd.Timestamp]
        """
        colors = plt.colormaps.get_cmap("rainbow")

        plots = []
        for idx, name in enumerate(series):
            df = self.dataframes[name].copy()
            if start_time and end_time:
                df = df[(df.index >= start_time) & (df.index <= end_time)]
            plots.append({
                "plot_type": "time_series",
                "data": df,
                "y": name,
                "label": name,
                "color": colors(idx / len(series))
            })

        self._plot(plots, plot_size=(12, 6),
                   fig_title=f"Time Series Comparison",
                   fig_xlabel="Time", fig_ylabel="Normalized Value")
