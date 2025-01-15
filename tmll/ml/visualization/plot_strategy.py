from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from matplotlib.axes import Axes


class PlotStrategy(ABC):
    """
    An abstract class to define the interface for different plot strategies.
    Each plot strategy will implement this interface based on the corresponding plot type.
    """

    @abstractmethod
    def plot(self, ax: Axes, data: Optional[Any] = None, **kwargs) -> None:
        """
        An abstract method to plot the data on the given axes.
        Each plot strategy will implement this method based on the corresponding plot type.

        :param ax: The axes to plot the data
        :type ax: Axes
        :param data: The data to plot
        :type Optional[Any]
        :param kwargs: Additional keyword arguments for the plot
        :return: None
        """
        pass

    @staticmethod
    def set_title_and_labels(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
        """
        Set the title and labels for the given axes.

        :param ax: The axes to set the title and labels
        :type ax: Axes
        :param title: Title of the plot
        :type title: str
        :param xlabel: Label for the x-axis
        :type xlabel: str
        :param ylabel: Label for the y-axis
        :type ylabel: str
        :return: None
        """

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def _get_x_y(self, data: Any, **kwargs) -> Tuple[Any, Any]:
        """
        Get the x and y values from the data based on the given keyword arguments.

        :param data: The data to extract the x and y values
        :type data: Any
        :param kwargs: Additional keyword arguments
        :return: The x and y values
        :rtype: tuple
        """
        if data is None:
            raise ValueError("Data is required for plotting")

        x = kwargs.get('x', None)
        y = kwargs.get('y', None)
        x = data[x] if x else data.index
        y = data[y] if y else data

        return x, y