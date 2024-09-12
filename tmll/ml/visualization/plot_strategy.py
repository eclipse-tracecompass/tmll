from abc import ABC, abstractmethod
from typing import Any

from matplotlib.axes import Axes


class PlotStrategy(ABC):
    """
    An abstract class to define the interface for different plot strategies.
    Each plot strategy will implement this interface based on the corresponding plot type.
    """

    @abstractmethod
    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        An abstract method to plot the data on the given axes.
        Each plot strategy will implement this method based on the corresponding plot type.

        Args:
            ax (Axes): The axes to plot the data.
            data (Any): The data to plot.
        """
        pass

    @staticmethod
    def set_title_and_labels(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
        """
        Set the title and labels for the given axes.

        Args:
            ax (Axes): The axes to set the title and labels.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
        """

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)