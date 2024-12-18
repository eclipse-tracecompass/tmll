from dataclasses import dataclass
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
from tmll.common.models.output import Output
from tmll.tmll_client import TMLLClient


# Minimum number of data points required for anomaly detection analysis to proceed with
MINIMUM_REQUIRED_DATAPOINTS = 10

# Anomaly detection methods supported by the module
DETECTION_METHODS = Literal["zscore", "iqr", "moving_average", "combined", "iforest", "seasonality", "frequency_domain"]


@dataclass
class AnomalyDetectionResult:
    anomalies: Dict[str, pd.DataFrame]
    anomaly_periods: Dict[str, List]


class AnomalyDetection(BaseModule):
    """
    A class for performing anomaly detection on time series data.

    This class implements various anomaly detection strategies and provides
    methods for data processing, anomaly detection, and result visualization.

    :param client: The TMLL client for data communication.
    :type client: TMLLClient
    """

    def __init__(self, client: TMLLClient, experiment: Experiment,
                 outputs: Optional[List[Output]] = None, **kwargs) -> None:
        """
        Initialize the AnomalyDetection module.

        :param client: The TMLL client for data communication.
        :type client: TMLLClient
        :param experiment: The experiment to analyze.
        :type experiment: Experiment
        :param outputs: The list of outputs to analyze.
        :type outputs: Optional[List[Output]]
        :param method: The anomaly detection method to use.
        :type method: str
        :param kwargs: Additional keyword arguments to pass to the anomaly detection method.
        :type kwargs: dict
        """
        super().__init__(client=client, experiment=experiment)

        self.detection_method: str = ""
        self.strategy_map: Dict[str, AnomalyDetectionStrategy] = {
            "zscore": ZScoreStrategy(),
            "iqr": IQRStrategy(),
            "moving_average": MovingAverageStrategy(),
            "combined": CombinedStrategy([ZScoreStrategy(), IQRStrategy(), MovingAverageStrategy()]),
            "iforest": IsolationForestStrategy(),
            "seasonality": SeasonalityStrategy(),
            "frequency_domain": FrequencyDomainStrategy()
        }

        self.logger.info(f"Initializing AnomalyDetection module")

        self._process(outputs, **kwargs)

    def _process(self, outputs: Optional[List[Output]] = None, **kwargs) -> None:
        super()._process(outputs=outputs,
                         min_size=kwargs.get("min_size", MINIMUM_REQUIRED_DATAPOINTS),
                         **kwargs)

    def _post_process(self, **kwargs) -> None:
        # Handle aggregation if requested
        if kwargs.get("aggregate", False):
            # Outer join all dataframes on timestamp
            self.dataframes["aggregated"] = reduce(lambda x, y: pd.merge(x, y, on="timestamp", how="outer"), self.dataframes.values())

    def find_anomalies(self, method: str = "zscore", **kwargs) -> Optional[AnomalyDetectionResult]:
        """
        Apply the specified anomaly detection method to the dataset.

        This internal method selects the appropriate strategy and applies it to the data.

        :param method: The anomaly detection method to use.
        :type method: str
        :param kwargs: Additional keyword arguments to pass to the detection method.
        :return: None
        """
        self.detection_method = method
        self.logger.info(f"Detecting anomalies using {self.detection_method} method...")

        strategy = self.strategy_map.get(self.detection_method)
        if not strategy:
            self.logger.error(f"Unknown detection method: {self.detection_method}")
            return None

        anomalies: Dict[str, pd.DataFrame] = {}
        anomaly_periods: Dict[str, List] = {}
        for output_key, dataframe in self.dataframes.items():
            anomalies[output_key], anomaly_periods[output_key] = strategy.detect_anomalies(dataframe.copy(), **kwargs)

        return AnomalyDetectionResult(anomalies=anomalies, anomaly_periods=anomaly_periods)

    def plot_anomalies(self, anomaly_detection_results: Optional[AnomalyDetectionResult] = None, **kwargs) -> None:
        """
        Plot the original dataframe features along with the anomaly periods.

        :param anomaly_detection_results: The anomaly detection results to plot.
        :type anomaly_detection_results: AnomalyDetectionResult
        :param kwargs: Additional keyword arguments for plotting
        :type kwargs: dict
        """
        if not anomaly_detection_results or not anomaly_detection_results.anomalies or not self.dataframes:
            self.logger.error("No data or anomalies detected.")
            return

        fig_size = kwargs.get("fig_size", (18, 4))
        fig_dpi = kwargs.get("fig_dpi", 500)
        colors = plt.colormaps.get_cmap("tab20")

        for output_key, dataframe in self.dataframes.items():
            # Check if the output key is in the anomalies
            if output_key not in anomaly_detection_results.anomalies:
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
                    "color": colors(dataframe.columns.get_loc(column) // len(dataframe.columns)),  # type: ignore
                    "alpha": 0.75
                })

            # Append the anomaly periods to the plots as span plot
            for start, end in anomaly_detection_results.anomaly_periods[output_key]:
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
            for point in anomaly_detection_results.anomalies[output_key].index:
                if not anomaly_detection_results.anomalies[output_key].loc[point].any():
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
                           plot_size=fig_size,
                           dpi=fig_dpi,
                           fig_title=f"Anomaly Detection for \"{output_name}\" using \"{self.detection_method.capitalize()}\" method",
                           fig_xlabel="Time (index)",
                           fig_ylabel="Normalized Values")
            else:
                for plot in plots:
                    self._plot([plot],
                               plot_size=fig_size,
                               dpi=fig_dpi,
                               fig_title=f"{plot["label"]}",
                               fig_xlabel="Time (index)",
                               fig_ylabel="Normalized Values")
