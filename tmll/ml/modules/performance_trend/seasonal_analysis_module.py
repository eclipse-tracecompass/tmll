from typing import Literal, Dict, Any, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt

from tmll.ml.modules.base_module import BaseModule
from tmll.tmll_client import TMLLClient
from tmll.ml.modules.common.data_fetch import DataFetcher
from tmll.ml.modules.common.data_preprocess import DataPreprocessor
from tmll.common.models.output import Output

TARGET_OUTPUTS = [
    Output.from_dict({
        "name": "Histogram",
        "id": "org.eclipse.tracecompass.internal.tmf.core.histogram.HistogramDataProvider",
        "type": "TREE_TIME_XY"
    })
]

class SeasonalAnalysis(BaseModule):
    """
    A module for performing seasonal analysis on time series data.

    This class provides methods for data fetching, preprocessing, and analyzing
    seasonal patterns in time series data. It includes functionality for
    automatic parameter optimization and various plotting capabilities.
    """

    def __init__(self, client: TMLLClient):
        """
        Initialize the SeasonalAnalysis module.

        :param client: The TMLL client for data access.
        :type client: TMLLClient
        """
        super().__init__(client=client)
        self.data_fetcher = DataFetcher(client)
        self.data_preprocessor = DataPreprocessor()
        self.dataframe = pd.DataFrame()
        self.analysis_results = {}
        self.optimal_params = {}

    def process(self, max_seasonality: Optional[int] = None, max_lag: Optional[int] = None, resample_frequency: Optional[str] = None, force_reload: bool = False, **kwargs) -> None:
        """
        Process the data for seasonal analysis with automatic parameter optimization.

        :param max_seasonality: The maximum seasonality to use for decomposition. If None, it will be optimized.
        :type max_seasonality: int, optional
        :param max_lag: The maximum lag for autocorrelation. If None, it will be optimized.
        :type max_lag: int, optional
        :param resample_frequency: The frequency to resample the data. If None, it will be optimized.
        :type resample_frequency: str, optional
        :param force_reload: If True, forces data reloading.
        :type force_reload: bool, optional
        :param kwargs: Additional keyword arguments for parameter optimization.
        """
        # Reset the analysis results
        self.analysis_results = {}
        self.optimal_params = {}

        if self.dataframe.empty or force_reload:
            self.dataframe = pd.DataFrame()

            self.logger.info("Fetching and preprocessing data...")
            data = self.data_fetcher.fetch_data(TARGET_OUTPUTS)
            if data is None:
                self.logger.error("No data fetched")
                return
            self.dataframe = self.data_preprocessor.normalize(data)
            self.dataframe['timestamp'] = pd.to_datetime(self.dataframe['timestamp'], unit='ns')
            self.dataframe.set_index('timestamp', inplace=True)

        if not resample_frequency:
            min_frequency = kwargs.get('min_resample_frequency', '25ms')
            resample_frequency = self._optimize_sampling_frequency(self.dataframe.index, min_frequency) # type: ignore
        self.logger.info(f"Resampling data to {resample_frequency}...")
        self.dataframe = self.data_preprocessor.resample(self.dataframe, frequency=resample_frequency)

        self.logger.info("Performing seasonal analysis...")
        for column in self.dataframe.columns:
            is_stationary = self._is_stationary(self.dataframe[column])
            
            method = kwargs.get('lag_method', 'aic')
            max_lag = self._optimize_max_lag(self.dataframe[column], max_lags=max_lag, method=method)
            
            # best_seasonality = self._optimize_seasonality(self.dataframe[column], max_seasonality)
            if max_seasonality:
                best_seasonality = min(max_seasonality, max_lag)
            best_seasonality = max_lag

            self.optimal_params[column] = {
                'seasonality': best_seasonality,
                'max_lag': max_lag
            }

            self.logger.info(f"Analyzing column {column} ({'non-' if not is_stationary else ''}stationary) with seasonality {best_seasonality} and max_lag {max_lag}...")
            
            self.analysis_results[column] = self._analyze_column(
                series=self.dataframe[column],
                seasonality=best_seasonality,
                max_lag=max_lag
            )

        self.logger.info("Seasonal analysis completed.")

    def _is_stationary(self, series: pd.Series, threshold: float = 0.05) -> bool:
        """
        Check if a time series is stationary using the Augmented Dickey-Fuller test.

        :param series: The time series to analyze.
        :type series: pd.Series
        :param threshold: The significance threshold for the test.
        :type threshold: float, optional
        :return: True if the series is stationary, False otherwise.
        :rtype: bool
        """
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series)
        if result[1] < threshold:
            return True
        
        return False

    def _optimize_sampling_frequency(self, timestamp_index: pd.DatetimeIndex, min_frequency: str = '25ms') -> str:
        """
        Optimize the sampling frequency based on the timestamp index.

        :param timestamp_index: The DatetimeIndex of the dataframe.
        :type timestamp_index: pd.DatetimeIndex
        :param min_frequency: The minimum allowed frequency.
        :type min_frequency: str, optional
        :return: Optimal sampling frequency as a string (e.g., '25ms', '1s', '1min').
        :rtype: str
        """
        deltas = timestamp_index.to_series().diff().dropna()
        mode_delta = deltas.mode()[0]
        
        if isinstance(mode_delta, pd.Timedelta):
            mode_delta_seconds = mode_delta.total_seconds()
            min_ms = pd.Timedelta(min_frequency).total_seconds()

            if mode_delta_seconds < min_ms:
                return '25ms'
            elif mode_delta_seconds < 1:
                return f'{max(int(mode_delta_seconds * 1000), 25)}ms'
            elif mode_delta_seconds < 60:
                return f'{int(mode_delta_seconds)}s'
            elif mode_delta_seconds < 3600:
                return f'{int(mode_delta_seconds // 60)}min'
            elif mode_delta_seconds < 86400:
                return f'{int(mode_delta_seconds // 3600)}h'
            else:
                return f'{int(mode_delta_seconds // 86400)}d'
        else:
            return '25ms'

    def _optimize_max_lag(self, series: pd.Series, method: Literal['aic', 'bic'] = 'aic', max_lags: Optional[int] = None) -> int:
        """
        Optimize the max_lag parameter for a given time series.

        :param series: The time series to analyze.
        :type series: pd.Series
        :param threshold: The significance threshold for PACF.
        :type threshold: float, optional
        :param min_lag: The minimum allowed lag.
        :type min_lag: int, optional
        :return: Optimal max_lag value.
        :rtype: int
        """
        # Convert series to numpy array if it's a pandas Series
        if isinstance(series, pd.Series):
            m_series = series.values
        
        n = len(m_series)
        
        # If max_lags is not provided, calculate it based on data size
        if max_lags is None:
            max_lags = min(int(np.ceil(10 * np.log10(n))), n // 2)
        
        # Ensure max_lags doesn't exceed data size
        max_lags = min(max_lags, n // 2)
        
        # Initialize best_lag and best_criterion
        best_lag = 0
        best_criterion = np.inf
        
        import warnings
        from statsmodels.tsa.ar_model import AutoReg
        method = 'aic'  # Use AIC as the criterion
        # Suppress warnings (optional, remove if you want to see all warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Iterate through possible lag values
            for lag in range(1, max_lags + 1):
                # Fit AutoRegressive model
                model = AutoReg(m_series, lags=lag) # type: ignore
                results = model.fit()
                
                # Calculate criterion (AIC or BIC)
                if method.lower() == 'aic':
                    criterion = results.aic
                else:
                    criterion = results.bic
                
                # Update best_lag if current lag gives better criterion
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_lag = lag
        
        return best_lag

    def _analyze_column(self, series: pd.Series, seasonality: int, max_lag: int, acf_threshold: float = 0.2) -> Dict[str, Any]:
        """
        Analyze a single column for seasonality.

        :param series: The time series to analyze.
        :type series: pd.Series
        :param seasonality: The seasonality to use for decomposition.
        :type seasonality: int
        :param max_lag: The maximum lag for autocorrelation.
        :type max_lag: int
        :param acf_threshold: The threshold for identifying potential seasons.
        :type acf_threshold: float, optional
        :return: A dictionary containing analysis results.
        :rtype: Dict[str, Any]
        """
        results = {}
        
        decomposition = seasonal_decompose(series, model='additive', period=max_lag)
        results['decomposition'] = decomposition
        
        detrended = series - decomposition.trend
        deseasonalized = series - decomposition.seasonal
        trend_strength = 1 - np.var(detrended) / np.var(series)
        seasonal_strength = 1 - np.var(deseasonalized) / np.var(detrended)
        results['trend_strength'] = trend_strength
        results['seasonal_strength'] = seasonal_strength
        
        acf_values = acf(series, nlags=max_lag)
        potential_seasons = [i for i in range(1, len(acf_values)) if acf_values[i] > acf_threshold]
        results['acf_values'] = acf_values
        results['potential_seasons'] = potential_seasons
        
        return results

    def plot(self, column: Optional[str] = None, plot_type: Literal['decomposition', 'acf', 'overview'] = 'overview') -> None:
        """
        Plot the analysis results.

        :param column: The specific column to plot. If None, plots all columns.
        :type column: str, optional
        :param plot_type: The type of plot to generate ('decomposition', 'acf', or 'overview').
        :type plot_type: str
        """
        if not self.analysis_results:
            self.logger.error("No analysis results available. Run process() first.")
            return

        columns_to_plot = [column] if column else list(self.analysis_results.keys())

        for col in columns_to_plot:
            if col not in self.analysis_results:
                self.logger.warning(f"No analysis results found for {col}")
                continue

            if plot_type == 'decomposition':
                self._plot_decomposition(col)
            elif plot_type == 'acf':
                self._plot_acf(col)
            elif plot_type == 'overview':
                self._plot_overview(col)
            else:
                self.logger.error(f"Unknown plot type: {plot_type}")

    def _plot_decomposition(self, column: str) -> None:
        """
        Plot the seasonal decomposition for a column.

        :param column: The column name to plot.
        :type column: str
        """
        decomposition = self.analysis_results[column]['decomposition']

        components = ['observed', 'trend', 'seasonal', 'resid']
        titles = [column, 'Trend', 'Seasonal', 'Residual']

        for component, title in zip(components, titles):
            plot_data = pd.DataFrame({
                'timestamp': getattr(decomposition, component).index,
                'value': getattr(decomposition, component).values
            })

            self._plot(plots = [{
                        'plot_type': 'time_series',
                        'data': plot_data,
                        'label': title,
                        'x': 'timestamp',
                        'y': 'value',
                        'color': 'blue',
                        'alpha': 0.7
                    }],
                    plot_size=(15, 3.5),
                    fig_title=title,
                    fig_xlabel='Time',
                    fig_ylabel='Value',
                    legend=False)

    def _plot_acf(self, column: str) -> None:
        """
        Plot the autocorrelation function for a column.

        :param column: The column name to plot.
        :type column: str
        """
        acf_values = self.analysis_results[column]['acf_values']
        potential_seasons = self.analysis_results[column]['potential_seasons']

        plot_data = pd.DataFrame({
            'lag': range(len(acf_values)),
            'acf': acf_values
        })

        plots = [{
            'plot_type': 'bar',
            'data': plot_data,
            'x': 'lag',
            'y': 'acf',
            'color': 'blue',
            'alpha': 0.7
        }]

        for season in potential_seasons:
            plots.append({
                'plot_type': 'vline',
                'data': None,
                'x': season,
                'color': 'red',
                'alpha': 0.5,
                'linestyle': '--'
            })

        self._plot(plots,
                   plot_size=(15, 6),
                   fig_title=f'Autocorrelation Function for {column}',
                   fig_xlabel='Lag',
                   fig_ylabel='Autocorrelation')

    def _plot_overview(self, column: str) -> None:
        """
        Plot an overview of the analysis results for a column.

        :param column: The column name to plot.
        :type column: str
        """
        results = self.analysis_results[column]
        decomposition = self.analysis_results[column]['decomposition']
        plots = []

        components = ['observed', 'trend', 'seasonal']
        labels = ['Original', 'Trend', 'Seasonal']
        colors = ['blue', 'orange', 'green']
        alphas = [0.6, 1, 0.8]

        for component, label, color, alpha in zip(components, labels, colors, alphas):
            plot_data = pd.DataFrame({
                'timestamp': getattr(decomposition, component).index,
                'value': getattr(decomposition, component).values
            })
            plots.append({
                'plot_type': 'time_series',
                'data': plot_data,
                'label': label,
                'x': 'timestamp',
                'y': 'value',
                'color': color,
                'alpha': alpha
            })

        idx = 0
        seasonality = self.optimal_params[column]['seasonality']
        max_idx = len(results['decomposition'].observed) // seasonality
        season_colors = plt.get_cmap('rainbow', max_idx)
        while True:
            if idx >= (max_idx - 1):
                break
            start = results['decomposition'].observed.index[idx * seasonality]
            end = results['decomposition'].observed.index[idx * seasonality + seasonality]
            plots.append({
                'plot_type': 'span',
                'data': None,
                'start': start,
                'end': end,
                'color': season_colors(idx / max_idx),
                'alpha': 0.25,
                'label': 'Seasons'
            })

            idx += 1

        self._plot(plots,
                   plot_size=(15, 3.5),
                   fig_title=f'Overview of {column}',
                   fig_xlabel='Time',
                   fig_ylabel='Value')