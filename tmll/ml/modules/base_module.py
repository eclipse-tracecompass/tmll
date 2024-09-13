from typing import Any, Dict, List, Literal, Tuple, Union
from abc import ABC, abstractmethod

import pandas as pd
import matplotlib.pyplot as plt

from tmll.common.models.output import Output
from tmll.common.services.logger import Logger
from tmll.ml.visualization.plot_factory import PlotFactory
from tmll.tmll_client import TMLLClient


# Variables
PLOT_TYPES = Literal[
    'time_series',
    'scatter',
    'histogram',
    'box',
    'violin',
    'heatmap',
    'pair',
    'bar'
]

class BaseModule(ABC):

    def __init__(self, client: TMLLClient):
        """
        Initialize the base module with the given TMLL client.

        :param client: The TMLL client to use
        :type client: TMLLClient
        """
        self.client = client
        
        self.logger = Logger(self.__class__.__name__)

    def _fetch_data(self, outputs: List[Output]) -> Union[None, Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]:
        """
        Fetch the data from the given outputs.

        :param outputs: The outputs to fetch the data from
        :type outputs: List[Output]
        :return: The fetched data
        :rtype: Union[None, Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]
        """
        
        self.client.fetch_outputs(custom_output_ids=[output.id for output in outputs])
        return self.client.fetch_data(custom_output_ids=[output.id for output in outputs], separate_columns=True)
    
    def _plot(self, plots: List[Dict[str, Any]], plot_size: Tuple[int, int] = (15, 10), **kwargs) -> None:
        """
        Plot the given plots. 

        :param plots: The plots to plot. Each plot should be a dictionary with the following keys:
            - plot_type (PLOT_TYPES): The type of the plot (e.g., 'time_series' or 'scatter')
            - data (Any): The data to plot
            - x (str): The name of the x-axis column
            - y (str): The name of the y-axis column
            - hue (str, optional): The name of the hue column. Default is None
            - color (str, optional): The color of the plot. Default is 'blue'
        :type plots: List[Dict[str, Any]]
        :param plot_size: The size of the plot. Default is (15, 10)
        :type plot_size: Tuple[int, int], optional
        :param kwargs: Additional keyword arguments to pass to the plot
            - title (str): The title of the plot
            - x (str): The name of the x-axis
            - y (str): The name of the y-axis
        :type kwargs: dict
        :return: None
        """

        fig, ax = plt.subplots(figsize=plot_size)

        for plot_info in plots:
            plot_type = plot_info['plot_type']
            data = plot_info['data']
            kwargs.update({k: v for k, v in plot_info.items() if k not in ['plot_type', 'data']})

            plot_strategy = PlotFactory.create_plot(plot_type)
            plot_strategy.plot(ax, data, **kwargs)

        ax.set_title(kwargs.get('title', ''))
        ax.set_xlabel(kwargs.get('x', ''))
        ax.set_ylabel(kwargs.get('y', ''))

        plt.tight_layout()
        plt.show()

    @abstractmethod
    def process(self):
        """
        An abstract method to process the module.
        Each concrete module should implement this method.

        :return: None
        """
        pass