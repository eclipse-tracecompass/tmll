import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Literal, Optional

from tmll.tmll_client import TMLLClient
from tmll.ml.modules.base_module import BaseModule
from tmll.ml.preprocess.normalizer import Normalizer
from tmll.common.models.output import Output


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

DETECTION_METHODS = Literal['zscore', 'iqr', 'moving_average', 'combined']

class AnomalyDetection(BaseModule):
    def __init__(self, client: TMLLClient):
        """
        Initialize the anomaly detection module.

        :param client: The TMLL client to use for fetching data
        :type client: TMLLClient
        """

        super().__init__(client=client)
        self.dataframe = pd.DataFrame()
        self.anomalies = {}

    def process(self, method: DETECTION_METHODS = 'combined', **kwargs) -> None:
        """
        Process the anomaly detection analysis using the specified method.
        Currently, the following outputs are being used to detect anomalies:
            - Histogram
            - System Call Latency - Latency vs Time
            - CPU Usage
            - Disk I/O View
            - IRQ Analysis - Latency vs Time
        The module fetches the data for these outputs (if present) and processes them to detect anomalies.

        :param method: The detection method to use. One of 'zscore', 'iqr', 'moving_average', 'combined'
        :type method: DETECTION_METHODS
        :param kwargs: Additional parameters for the detection method(s)
        :type kwargs: dict
        :return: None
        """

        self.logger.info(f"Starting anomaly detection analysis using {method} method...")

        self.logger.info("Fetching data...")
        data = self._fetch_data(outputs=TARGET_OUTPUTS[:])
        if not data:
            self.logger.error("No data fetched")
            return

        final_dataframe = pd.DataFrame()
        for output in TARGET_OUTPUTS[:]:
            if output.id not in data:
                self.logger.warning(f"The trace data does not contain the output {output.name}.")
                continue

            self.logger.info(f"Processing output {output.name}.")

            if isinstance(data[output.id], dict):
                for _, value in data[output.id].items():
                    if isinstance(value, pd.DataFrame):
                        dataframe: pd.DataFrame = value
                        dataframe = dataframe.rename(columns={'y': output.name, 'x': 'timestamp'})
                        if final_dataframe.empty:
                            final_dataframe = dataframe
                        else:
                            final_dataframe = pd.merge(final_dataframe, dataframe, on='timestamp', how='outer')
            else:
                continue

        # Since some datapoints may be missing, we need to fill them with 0
        final_dataframe = final_dataframe.fillna(0)

        # We normalize the data
        normalization_method = kwargs.get('normalization_method', 'minmax')
        normalizer = Normalizer(dataset=final_dataframe, method=normalization_method)
        final_dataframe = normalizer.normalize(target_features=[col for col in final_dataframe.columns if col != 'timestamp'])

        self.dataframe = final_dataframe

        self._detect_anomalies(detection_method=method, **kwargs)

    def _detect_anomalies(self, detection_method: str, **kwargs) -> None:
        """
        Detect anomalies in the data using the specified detection method.

        :param detection_method: The detection method to use. One of 'zscore', 'iqr', 'moving_average', 'combined'
        :type detection_method: str
        :param kwargs: Additional parameters for the detection method(s)
        :type kwargs: dict
        :return: None
        """

        self.logger.info(f"Detecting anomalies using {detection_method} method...")

        for column in self.dataframe.columns:
            if column != 'timestamp':
                method_func = self._get_detection_method(detection_method)
                if method_func:
                    self.anomalies[column] = method_func(column, **kwargs)

    def _get_detection_method(self, detection_method: str):
        """
        Returns the appropriate anomaly detection function based on the detection method.

        :param detection_method: The detection method to use. One of 'zscore', 'iqr', 'moving_average', 'combined'
        :type detection_method: str
        :return: The anomaly detection function
        :rtype: callable
        """

        method_map = {
            'zscore': self._detect_anomalies_zscore,
            'iqr': self._detect_anomalies_iqr,
            'moving_average': self._detect_anomalies_moving_average,
            'combined': self._detect_anomalies_combined
        }
        return method_map.get(detection_method)

    def _detect_anomalies_zscore(self, column: str, **kwargs) -> pd.DataFrame:
        """
        Detect anomalies using the Z-score method.

        :param column: The column to detect anomalies on
        :type column: str
        :param kwargs: Additional parameters for the Z-score method
        :type kwargs: dict
        :return: DataFrame with detected anomalies
        :rtype: pd.DataFrame
        """

        threshold = kwargs.get('zscore_threshold', 4)
        z_scores = np.abs(stats.zscore(self.dataframe[column]))
        return self.dataframe[z_scores > threshold][['timestamp', column]]

    def _detect_anomalies_iqr(self, column: str, **kwargs) -> pd.DataFrame:
        """
        Detect anomalies using the interquartile range (IQR) method.

        :param column: The column to detect anomalies on
        :type column: str
        :param kwargs: Additional parameters for the IQR method
        :type kwargs: dict
        :return: DataFrame with detected anomalies
        :rtype: pd.DataFrame
        """

        Q1 = self.dataframe[column].quantile(0.25)
        Q3 = self.dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.dataframe[(self.dataframe[column] < lower_bound) | (self.dataframe[column] > upper_bound)][['timestamp', column]]

    def _detect_anomalies_moving_average(self, column: str, **kwargs) -> pd.DataFrame:
        """
        Detect anomalies using the moving average method.

        :param column: The column to detect anomalies on
        :type column: str
        :param kwargs: Additional parameters for the moving average method
        :type kwargs: dict
        :return: DataFrame with detected anomalies
        :rtype: pd.DataFrame
        """

        window_size = kwargs.get('moving_average_window_size', 10)
        threshold = kwargs.get('moving_average_threshold', 0.1)
        moving_avg = self.dataframe[column].rolling(window=window_size).mean()
        return self.dataframe[np.abs(self.dataframe[column] - moving_avg) > threshold][['timestamp', column]]

    def _detect_anomalies_combined(self, column: str, methods: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """
        Detect anomalies by combining multiple detection methods dynamically.
        
        :param column: The column to detect anomalies on
        :type column: str
        :param methods: List of detection method names to combine (e.g., ['zscore', 'iqr', 'moving_average'])
        :type methods: Optional[List[str]]
        :param kwargs: Additional parameters for each detection method
        :type kwargs: dict
        :return: DataFrame with combined anomalies
        :rtype: pd.DataFrame
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'moving_average']

        anomaly_dataframes = []
        
        for method in methods:
            detection_func = self._get_detection_method(method)
            if detection_func:
                anomalies = detection_func(column, **kwargs)
                anomaly_dataframes.append(anomalies)
        
        if not anomaly_dataframes:
            self.logger.warning(f"No anomalies detected for column {column} using the specified methods.")
            return pd.DataFrame(columns=['timestamp', column])

        # Perform intersection of anomalies based on 'timestamp'
        combined_anomalies = anomaly_dataframes[0]
        for anomalies in anomaly_dataframes[1:]:
            combined_anomalies = pd.merge(combined_anomalies, anomalies, on='timestamp', how='inner')

        # Deduplicate column names and select the final column
        combined_anomalies[column] = combined_anomalies[[col for col in combined_anomalies.columns if column in col]].iloc[:, 0]

        return combined_anomalies[['timestamp', column]]


    def plot(self, columns: Optional[List[str]] = None) -> None:
        """
        Plot the original data and detected anomalies for the specified columns.

        :param columns: List of columns to plot. Default is all columns except 'timestamp'
        :type columns: Optional[List[str]]
        :return: None
        """

        if columns is None:
            columns = [col for col in self.dataframe.columns if col != 'timestamp']

        num_plots = len(columns)
        total_height = num_plots
        
        for column in columns:
            plot_data = self.dataframe[['timestamp', column]].copy()
            
            plot_dict = {
                'original': plot_data,
                'anomalies': pd.DataFrame()
            }
            
            if column in self.anomalies:
                plot_dict['anomalies'] = self.anomalies[column]

            self._plot(
                plots=[{'plot_type': 'time_series', 'data': plot_dict['original'], 'color': 'blue'},
                       {'plot_type': 'scatter', 'data': plot_dict['anomalies'], 'color': 'red'}],
                plot_size=(15, total_height),
                x='timestamp',
                y=column,
                title=f'Anomaly Detection for {column}'
            )