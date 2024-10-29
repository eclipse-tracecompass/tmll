import pandas as pd
from typing import List, Literal, Dict, Optional
import matplotlib.pyplot as plt

from functools import reduce

from tmll.common.models.experiment import Experiment
from tmll.ml.modules.anomaly_detection.strategies.base import AnomalyDetectionStrategy
from tmll.ml.modules.anomaly_detection.strategies.frequency_domain import FrequencyDomainStrategy
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


# Minimum number of data points required for anomaly detection analysis to proceed with
MINIMUM_REQUIRED_DATAPOINTS = 10

# Anomaly detection methods supported by the module
DETECTION_METHODS = Literal["zscore", "iqr", "moving_average", "combined", "iforest", "seasonality", "frequency_domain"]

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
        self.experiment: Experiment = Experiment("", "", 0, 0, 0, "")
        self.detection_method: str = ""
        self.data_fetcher: DataFetcher = DataFetcher(client)
        self.data_preprocessor: DataPreprocessor = DataPreprocessor()
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.anomalies: Dict[str, pd.DataFrame] = {}
        self.anomaly_periods: Dict[str, List] = {}
        self.strategy_map: Dict[str, AnomalyDetectionStrategy] = {
            "zscore": ZScoreStrategy(),
            "iqr": IQRStrategy(),
            "moving_average": MovingAverageStrategy(),
            "combined": CombinedStrategy([ZScoreStrategy(), IQRStrategy(), MovingAverageStrategy()]),
            "iforest": IsolationForestStrategy(),
            "seasonality": SeasonalityStrategy(),
            "frequency_domain": FrequencyDomainStrategy()
        }

    def process(self, experiment: Experiment, outputs: Optional[List[Output]] = None, method: str = "zscore", aggregate: bool = False, force_reload: bool = False, **kwargs) -> None:
        """
        Process the data and perform anomaly detection.
        This method fetches data if necessary, preprocesses it, and applies the specified
        anomaly detection method.

        :param experiment: The experiment to process.
        :type experiment: Experiment
        :param outputs: The list of output IDs to process. If None, all outputs are processed.
        :type outputs: List[int], optional
        :param method: The anomaly detection method to use.
        :type method: str
        :param aggregate: Whether to aggregate the dataframes into a single dataframe.
        :type aggregate: bool
        :param force_reload: Whether to force reload the data.
        :type force_reload: bool
        :param kwargs: Additional keyword arguments to pass to the anomaly detection method.
        :return: None
        """
        # Reset the anomalies
        self.anomalies.clear()
        self.anomaly_periods.clear()

        self.experiment = experiment

        if force_reload or not self.dataframes:
            self.logger.info(f"Starting anomaly detection analysis using {method} method...")
            
            self.dataframes.clear()

            data = self.data_fetcher.fetch_data(experiment=experiment, target_outputs=outputs)
            if data is None:
                self.logger.error("No data fetched")
                return
            
            for output_key, output_data in data.items():
                output_type = next((output.type for output in (outputs or []) if str(output.id) in output_key), None)

                if output_type and output_type == "TIME_GRAPH":
                    output_data["duration"] = output_data["end_time"] - output_data["start_time"]

                    output_data["timestamp"] = output_data["start_time"]

                    # Drop start_time and end_time columns
                    output_data.drop(columns=["start_time", "end_time", "entry_id"], inplace=True, errors="ignore")

                    # Separate the data into multiple dataframes
                    separated_dataframes = self.data_preprocessor.separate_timegraph(output_data, "label")
                    for key, value in separated_dataframes.items():
                        # If the dataframe doesn"t have any instances, skip it
                        if value.empty:
                            continue

                        self.dataframes[f"{output_key}${key}"] = value
                else:
                    self.dataframes[output_key] = output_data

            resample_freq = kwargs.get("resample_freq", None)
            if not resample_freq:
                self.logger.warning("No resample frequency provided. Using default frequency of 1 second.")
                resample_freq = "1s"

            for output_key, output_data in self.dataframes.items():
                self.dataframes[output_key] = self.data_preprocessor.normalize(output_data)
                self.dataframes[output_key] = self.data_preprocessor.convert_to_datetime(self.dataframes[output_key])
                self.dataframes[output_key] = self.data_preprocessor.resample(self.dataframes[output_key], frequency=resample_freq)

            # Remove dataframes with less than the minimum required data points
            keys = list(self.dataframes.keys())
            for key in keys:
                if self.dataframes[key].shape[0] < MINIMUM_REQUIRED_DATAPOINTS:
                    del self.dataframes[key]

            if aggregate:
                # Outer join all the dataframes on the timestamp column into a single dataframe
                self.dataframes["aggregated"] = reduce(lambda x, y: pd.merge(x, y, on="timestamp", how="outer"), self.dataframes.values())

                # Remove the individual dataframes
                keys = list(self.dataframes.keys())
                for key in keys:
                    if key != "aggregated":
                        del self.dataframes[key]

        self.detection_method = method
        self._detect_anomalies(**kwargs)

    def _detect_anomalies(self, **kwargs) -> None:
        """
        Apply the specified anomaly detection method to the dataset.

        This internal method selects the appropriate strategy and applies it to the data.

        :param detection_method: The name of the detection method to use.
        :type detection_method: str
        :param kwargs: Additional keyword arguments to pass to the detection method.
        :return: None
        """
        self.logger.info(f"Detecting anomalies using {self.detection_method} method...")

        strategy = self.strategy_map.get(self.detection_method)
        if not strategy:
            self.logger.error(f"Unknown detection method: {self.detection_method}")
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
        
        colors = plt.colormaps.get_cmap("tab20")

        for output_key, dataframe in self.dataframes.items():
            # Check if the output key is in the anomalies
            if output_key not in self.anomalies:
                continue

            plots = []
            # Plot the original data
            for column in dataframe.columns:
                # Plot_data is a DataFrame with the index and the column data
                plot_data = dataframe[[column]].copy()
                plot_data["timestamp"] = plot_data.index

                plots.append({
                    "plot_type": "time_series",
                    "data": plot_data,
                    "label": column,
                    "x": "timestamp",
                    "y": column,
                    "color": colors(dataframe.columns.get_loc(column) // len(dataframe.columns)), # type: ignore
                    "alpha": 0.75
                })

            # Append the anomaly periods to the plots as span plot
            for start, end in self.anomaly_periods[output_key]:
                # print(f"Anomaly detected from {start} to {end}")
                plots.append({
                    "label": "Anomaly Period",
                    "plot_type": "span",
                    "data": None,
                    "start": start,
                    "end": end,
                    "color": "pink",
                    "alpha": 0.5,
                    "is_top": True
                })

            anomaly_points_list = []
            for point in self.anomalies[output_key].index:
                if not self.anomalies[output_key].loc[point].any():
                    continue
                
                anomaly_points_list.append({
                    "timestamp": point,
                    "value": dataframe.loc[point].values[0],
                })
                
            if anomaly_points_list:
                # Create the DataFrame after the loop
                anomaly_points = pd.DataFrame(anomaly_points_list)
                plots.append({
                    "plot_type": "scatter",
                    "data": anomaly_points,
                    "label": "Anomaly Points",
                    "x": "timestamp",
                    "y": "value",
                    "color": "red",
                    "alpha": 0.75
                })

            output_name = ""
            for output in self.experiment.outputs:
                if str(output.id) in output_key:
                    output_name = output.name
                    if "$" in output_key:
                        output_name += f" ({output_key.split("$")[1]})"
                    break

            is_separate = kwargs.get("separate", False)
            if not is_separate:
                self._plot(plots,
                    plot_size=(18, 4),
                        dpi=500,
                    fig_title=f"Anomaly Detection for \"{output_name}\" using \"{self.detection_method.capitalize()}\" method",
                    fig_xlabel="Time (index)",
                    fig_ylabel="Normalized Values")
            else:
                for plot in plots:
                    self._plot([plot],
                            plot_size=(18, 3),
                            dpi=500,
                            fig_title=f"{plot["label"]}",
                            fig_xlabel="Time (index)",
                            fig_ylabel="Normalized Values")