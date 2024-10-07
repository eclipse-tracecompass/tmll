import pandas as pd
from typing import List, Literal, Dict

from tmll.ml.modules.anomaly_detection.strategies.base import AnomalyDetectionStrategy
from tmll.ml.modules.base_module import BaseModule
from tmll.ml.modules.anomaly_detection.strategies.combined import CombinedStrategy
from tmll.ml.modules.anomaly_detection.strategies.iqr import IQRStrategy
from tmll.ml.modules.anomaly_detection.strategies.iforest import IsolationForestStrategy
from tmll.ml.modules.anomaly_detection.strategies.moving_average import MovingAverageStrategy
from tmll.ml.modules.anomaly_detection.strategies.zscore import ZScoreStrategy
from tmll.ml.modules.anomaly_detection.strategies.seasonality import SeasonalityStrategy
from tmll.ml.modules.common.data_fetch import DataFetcher
from tmll.ml.modules.common.data_preprocess import DataPreprocessor
from tmll.common.models.output import Output
from tmll.tmll_client import TMLLClient


TARGET_OUTPUTS = [
    Output.from_dict({
        "name": "Histogram",
        "id": "org.eclipse.tracecompass.internal.tmf.core.histogram.HistogramDataProvider",
        "type": "TREE_TIME_XY"
    }),
    Output.from_dict({
        "name": "System Call Latency - Latency vs Time",
        "id": ("org.eclipse.tracecompass.internal.analysis.timing.core.segmentstore.scatter.dataprovider:",
               "org.eclipse.tracecompass.analysis.os.linux.latency.syscall"),
        "type": "TREE_TIME_XY"
    }),
    Output.from_dict({
        "name": "CPU Usage",
        "id": "org.eclipse.tracecompass.analysis.os.linux.core.cpuusage.CpuUsageDataProvider",
        "type": "TREE_TIME_XY"
    }),
    Output.from_dict({
        "name": "Disk I/O View",
        "id": "org.eclipse.tracecompass.analysis.os.linux.core.inputoutput.DisksIODataProvider",
        "type": "TREE_TIME_XY"
    }),
    # Output.from_dict({
    #     "name": "IRQ Analysis - Latency vs Time",
    #     "id": "org.eclipse.tracecompass.internal.analysis.timing.core.segmentstore.scatter.dataprovider:lttng.analysis.irq",
    #     "type": "TREE_TIME_XY"
    # }),
]

DETECTION_METHODS = Literal['zscore', 'iqr', 'moving_average', 'combined', 'iforest', 'seasonality']

class AnomalyDetection(BaseModule):
    """
    A class for performing anomaly detection on time series data.

    This class implements various anomaly detection strategies and provides
    methods for data processing, anomaly detection, and result visualization.

    :param client: The TMLL client for data communication.
    :type client: TMLLClient
    """

    def __init__(self, client: TMLLClient):
        """
        Initialize the AnomalyDetection module.

        :param client: The TMLL client for data communication.
        :type client: TMLLClient
        """
        super().__init__(client=client)
        self.data_fetcher = DataFetcher(client)
        self.data_preprocessor = DataPreprocessor()
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.anomalies: Dict[str, pd.DataFrame] = {}
        self.anomaly_periods: Dict[str, List] = {}
        self.strategy_map: Dict[str, AnomalyDetectionStrategy] = {
            'zscore': ZScoreStrategy(),
            'iqr': IQRStrategy(),
            'moving_average': MovingAverageStrategy(),
            'combined': CombinedStrategy([ZScoreStrategy(), IQRStrategy(), MovingAverageStrategy()]),
            'iforest': IsolationForestStrategy(),
            'seasonality': SeasonalityStrategy()
        }

    def process(self, method: str = 'iforest', aggregate: bool = True, force_reload: bool = False, **kwargs) -> None:
        """
        Process the data and perform anomaly detection.

        This method fetches data if necessary, preprocesses it, and applies the specified
        anomaly detection method.

        :param method: The anomaly detection method to use.
        :type method: str, optional
        :param aggregate: If True, aggregate all the outputs into a single dataframe.
        :type aggregate: bool, optional
        :param force_reload: If True, forces data reloading.
        :type force_reload: bool, optional
        :param kwargs: Additional keyword arguments to pass to the anomaly detection method.
        :return: None
        """
        # Reset the anomalies
        self.anomalies.clear()
        self.anomaly_periods.clear()

        if force_reload or not self.dataframes:
            self.logger.info(f"Starting anomaly detection analysis using {method} method...")
            
            self.dataframes.clear()

            data = self.data_fetcher.fetch_data(TARGET_OUTPUTS)
            if data is None:
                self.logger.error("No data fetched")
                return
            
            for output_key, output_data in data.items():
                self.dataframes[output_key] = self.data_preprocessor.normalize(output_data)
                self.dataframes[output_key] = self.data_preprocessor.convert_to_datetime(self.dataframes[output_key])

            if aggregate:
                from functools import reduce
                # Outer join all the dataframes on the timestamp column into a single dataframe
                self.dataframes['aggregated'] = reduce(lambda x, y: pd.merge(x, y, on='timestamp', how='outer'), self.dataframes.values())

                # Remove the individual dataframes
                keys = list(self.dataframes.keys())
                for key in keys:
                    if key != 'aggregated':
                        del self.dataframes[key]

        self._detect_anomalies(detection_method=method, **kwargs)

    def _detect_anomalies(self, detection_method: str, **kwargs) -> None:
        """
        Apply the specified anomaly detection method to the dataset.

        This internal method selects the appropriate strategy and applies it to the data.

        :param detection_method: The name of the detection method to use.
        :type detection_method: str
        :param kwargs: Additional keyword arguments to pass to the detection method.
        :return: None
        """
        self.logger.info(f"Detecting anomalies using {detection_method} method...")

        strategy = self.strategy_map.get(detection_method)
        if not strategy:
            self.logger.error(f"Unknown detection method: {detection_method}")
            return
        
        for output_key, dataframe in self.dataframes.items():
            self.anomalies[output_key], self.anomaly_periods[output_key] = strategy.detect_anomalies(dataframe.copy(), **kwargs)

    def plot(self, **kwargs) -> None:
        """
        Plot the original dataframe features along with the anomaly periods.

        This method creates a visualization of the original data and highlights
        the detected anomaly periods.

        :return: None
        """
        if not self.dataframes or not self.anomalies:
            self.logger.error("No data or anomalies detected.")
            return
        
        import matplotlib.pyplot as plt
        colors = plt.colormaps.get_cmap('tab20')

        for output_key, dataframe in self.dataframes.items():
            plots = []
            # Plot the original data
            for column in dataframe.columns:
                # Plot_data is a DataFrame with the index and the column data
                plot_data = dataframe[[column]].copy()
                plot_data['timestamp'] = plot_data.index

                plots.append({
                    'plot_type': 'time_series',
                    'data': plot_data,
                    'label': column,
                    'x': 'timestamp',
                    'y': column,
                    'color': colors(dataframe.columns.get_loc(column) // len(dataframe.columns)), # type: ignore
                    'alpha': 0.75
                })

            # Append the anomaly periods to the plots as span plot
            for start, end in self.anomaly_periods[output_key]:
                self_time = pd.to_datetime(dataframe.iloc[start].index, unit='ns') # type: ignore
                end_time = pd.to_datetime(dataframe.iloc[end].index, unit='ns') # type: ignore
                print(f"Anomaly detected from {self_time} to {end_time}")
                plots.append({
                    'label': 'Anomaly Period',
                    'plot_type': 'span',
                    'data': None,
                    'start': start,
                    'end': end,
                    'color': 'red',
                    'alpha': 0.5,
                    'is_top': True
                })

            is_separate = kwargs.get('separate', False)
            if not is_separate:
                self._plot(plots,
                    plot_size=(18, 4),
                        dpi=500,
                    fig_title='Anomaly Detection - Isolation Forest',
                    fig_xlabel='Time (index)',
                    fig_ylabel='Normalized Values')
            else:
                for plot in plots:
                    self._plot([plot],
                            plot_size=(18, 3),
                            dpi=500,
                            fig_title=f'{plot["label"]}',
                            fig_xlabel='Time (index)',
                            fig_ylabel='Normalized Values')