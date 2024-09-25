import pandas as pd
from typing import Literal

from tmll.ml.modules.base_module import BaseModule
from tmll.ml.modules.anomaly_detection.strategies.combined import CombinedStrategy
from tmll.ml.modules.anomaly_detection.strategies.iqr import IQRStrategy
from tmll.ml.modules.anomaly_detection.strategies.iforest import IsolationForestStrategy
from tmll.ml.modules.anomaly_detection.strategies.moving_average import MovingAverageStrategy
from tmll.ml.modules.anomaly_detection.strategies.zscore import ZScoreStrategy
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
    Output.from_dict({
        "name": "IRQ Analysis - Latency vs Time",
        "id": "org.eclipse.tracecompass.internal.analysis.timing.core.segmentstore.scatter.dataprovider:lttng.analysis.irq",
        "type": "TREE_TIME_XY"
    }),
]

DETECTION_METHODS = Literal['zscore', 'iqr', 'moving_average', 'combined', 'iforest']

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
        self.dataframe = pd.DataFrame()
        self.anomalies = pd.DataFrame()
        self.anomaly_periods = []
        self.strategy_map = {
            'zscore': ZScoreStrategy(),
            'iqr': IQRStrategy(),
            'moving_average': MovingAverageStrategy(),
            'combined': CombinedStrategy([ZScoreStrategy(), IQRStrategy(), MovingAverageStrategy()]),
            'iforest': IsolationForestStrategy()
        }

    def process(self, method: str = 'iforest', force_reload: bool = False, **kwargs) -> None:
        """
        Process the data and perform anomaly detection.

        This method fetches data if necessary, preprocesses it, and applies the specified
        anomaly detection method.

        :param method: The anomaly detection method to use.
        :type method: str, optional
        :param force_reload: If True, forces data reloading.
        :type force_reload: bool, optional
        :param kwargs: Additional keyword arguments to pass to the anomaly detection method.
        :return: None
        """
        # Reset the anomalies
        self.anomalies = pd.DataFrame()
        self.anomaly_periods = []

        if self.dataframe.empty or force_reload:
            self.logger.info(f"Starting anomaly detection analysis using {method} method...")
            data = self.data_fetcher.fetch_data(TARGET_OUTPUTS)
            if data is None:
                self.logger.error("No data fetched")
                return
            
            self.dataframe = self.data_preprocessor.normalize(data)

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

        self.anomalies, self.anomaly_periods = strategy.detect_anomalies(self.dataframe, **kwargs)

    def plot(self):
        """
        Plot the original dataframe features along with the anomaly periods.

        This method creates a visualization of the original data and highlights
        the detected anomaly periods.

        :return: None
        """
        if self.dataframe.empty or self.anomalies.empty:
            self.logger.error("No data or anomalies detected.")
            return
        
        colors = {
            'Histogram': 'blue',
            'System Call Latency - Latency vs Time': 'cyan',
            'CPU Usage': 'pink',
            'Disk I/O View': 'orange',
            'IRQ Analysis - Latency vs Time': 'limegreen'
        }

        plots = []
        # Plot the original data
        for column in self.dataframe.columns:
            if column != 'timestamp':
                # Plot_data is a DataFrame with the index and the column data
                plot_data = self.dataframe[['timestamp', column]].copy()
                plot_data['timestamp'] = plot_data.index

                plots.append({
                    'plot_type': 'time_series',
                    'data': plot_data,
                    'label': column,
                    'x': 'timestamp',
                    'y': column,
                    'color': colors.get(column, 'blue'),
                    'alpha': 0.6
                })

        # Append the anomaly periods to the plots as span plot
        for start, end in self.anomaly_periods:
            plots.append({
                'label': 'Anomaly Period',
                'plot_type': 'span',
                'data': None,
                'start': start,
                'end': end,
                'color': 'red',
                'alpha': 0.75,
                'is_top': True
            })

        self._plot(plots,
                   plot_size=(15, 3.5),
                   fig_title='Anomaly Detection - Isolation Forest',
                   fig_xlabel='Time (index)',
                   fig_ylabel='Normalized Values')