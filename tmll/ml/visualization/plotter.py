from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt

from tmll.ml.visualization.plot_factory import PlotFactory


class Plotter:
    """
    A class to plot the data based on the given plot configurations.
    It uses the PlotFactory to create the plot strategies based on the plot types, and then plots the data.
    """

    @staticmethod
    def plot(data, plot_configs: List[Dict[str, Any]], figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot the data based on the given plot configurations.

        Args:
            data: The data to plot.
            plot_configs: A list of dictionaries containing the plot configurations.
            figsize: The size of the figure to display the plots. Default is (15, 10).
        """

        num_plots = len(plot_configs)
        fig, axs = plt.subplots(num_plots, 1, figsize=figsize)
        if num_plots == 1:
            axs = [axs]

        for ax, config in zip(axs, plot_configs):
            plot_type = config.pop('type')
            plot_strategy = PlotFactory.create_plot(plot_type)
            plot_strategy.plot(ax, data, **config)

        plt.tight_layout()
        plt.show()