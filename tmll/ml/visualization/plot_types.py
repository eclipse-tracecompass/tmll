from typing import Any

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from tmll.ml.visualization.plot_strategy import PlotStrategy


class TimeSeriesPlot(PlotStrategy):
    """A concrete class to implement the time series plot strategy."""

    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the time series data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The time series data to plot.
        :type data: Any
        :param x: The name of the x-axis column. Default is 'timestamp'.
        :type x: str, optional
        :param y: The name of the y-axis column.
        :type y: str
        :param hue: The name of the hue column. Default is None.
        :type hue: str, optional
        :param color: The color of the plot. Default is 'blue'.
        :type color: str, optional
        :return: None
        """
        x = kwargs.get('x', 'timestamp')
        y = kwargs.get('y')
        hue = kwargs.get('hue', None)
        color = kwargs.get('color', 'blue')
        sns.lineplot(data=data, x=x, y=y, hue=hue, ax=ax, color=color)
        self.set_title_and_labels(ax, f"Time Series of {y}", str(x), str(y))


class ScatterPlot(PlotStrategy):
    """A concrete class to implement the scatter plot strategy."""

    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the scatter plot of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any
        :param x: The name of the x-axis column.
        :type x: str
        :param y: The name of the y-axis column.
        :type y: str
        :param hue: The name of the hue column. Default is None.
        :type hue: str, optional
        :param color: The color of the plot. Default is 'blue'.
        :type color: str, optional
        :return: None
        """
        x = kwargs.get('x')
        y = kwargs.get('y')
        hue = kwargs.get('hue', None)
        color = kwargs.get('color', 'blue')
        sns.scatterplot(data=data, x=x, y=y, ax=ax, color=color, hue=hue)
        self.set_title_and_labels(ax, f"Scatter Plot of {y} vs {x}", str(x), str(y))


class HistogramPlot(PlotStrategy):
    """A concrete class to implement the histogram plot strategy."""

    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the histogram of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any
        :param column: The name of the column to plot.
        :type column: str
        :param color: The color of the plot. Default is 'blue'.
        :type color: str, optional
        :param bins: The number of bins for the histogram. Default is 50.
        :type bins: int, optional
        :return: None
        """
        column = kwargs.get('column')
        color = kwargs.get('color', 'blue')
        bins = kwargs.get('bins', 50)
        if isinstance(bins, dict):
            bins = bins.get('bins', 50)
        sns.histplot(data=data, x=column, bins=bins, ax=ax, kde=True, color=color)
        self.set_title_and_labels(ax, f"Histogram of {column}", str(column), "Frequency")


class BoxPlot(PlotStrategy):
    """A concrete class to implement the box plot strategy."""

    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the box plot of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any
        :param x: The name of the x-axis column.
        :type x: str
        :param y: The name of the y-axis column.
        :type y: str
        :param color: The color of the plot. Default is 'blue'.
        :type color: str, optional
        :return: None
        """
        x = kwargs.get('x')
        y = kwargs.get('y')
        color = kwargs.get('color', 'blue')
        sns.boxplot(data=data, x=x, y=y, ax=ax, color=color)
        self.set_title_and_labels(ax, f"Box Plot of {y} by {x}", str(x), str(y))


class ViolinPlot(PlotStrategy):
    """A concrete class to implement the violin plot strategy."""

    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the violin plot of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any
        :param x: The name of the x-axis column.
        :type x: str
        :param y: The name of the y-axis column.
        :type y: str
        :param color: The color of the plot. Default is 'blue'.
        :type color: str, optional
        :return: None
        """
        x = kwargs.get('x')
        y = kwargs.get('y')
        color = kwargs.get('color', 'blue')
        sns.violinplot(data=data, x=x, y=y, ax=ax, color=color)
        self.set_title_and_labels(ax, f"Violin Plot of {y} by {x}", str(x), str(y))


class HeatmapPlot(PlotStrategy):
    """A concrete class to implement the heatmap plot strategy."""
    
    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the heatmap of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any
        :param cmap: The name of the colormap. Default is 'viridis'.
        :type cmap: str, optional
        :param annot: Whether to annotate the heatmap. Default is True.
        :type annot: bool, optional
        :param fmt: The format of the annotation. Default is '.2f'.
        :type fmt: str, optional
        :return: None
        """
        cmap = kwargs.get('cmap', 'viridis')
        annot = kwargs.get('annot', True)
        fmt = kwargs.get('fmt', '.2f')
        sns.heatmap(data, ax=ax, cmap=cmap, annot=annot, fmt=fmt)
        self.set_title_and_labels(ax, "Correlation Heatmap", "", "")


class PairPlot(PlotStrategy):
    """A concrete class to implement the pair plot strategy."""

    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the pair plot of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any
        :param vars: The list of variables to plot.
        :type vars: list
        :param hue: The name of the hue column.
        :type hue: str
        :return: None
        """
        vars = kwargs.get('vars')
        hue = kwargs.get('hue')
        sns.pairplot(data=data, vars=vars, hue=str(hue))
        plt.suptitle("Pair Plot", y=1.02)


class BarPlot(PlotStrategy):
    """A concrete class to implement the bar plot strategy."""

    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the bar plot of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any
        :param x: The name of the x-axis column.
        :type x: str
        :param y: The name of the y-axis column.
        :type y: str
        :param hue: The name of the hue column. Default is None.
        :type hue: str, optional
        :param color: The color of the plot. Default is 'blue'.
        :type color: str, optional
        :return: None
        """
        x = kwargs.get('x')
        y = kwargs.get('y')
        hue = kwargs.get('hue', None)
        color = kwargs.get('color', 'blue')
        sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, color=color)
        self.set_title_and_labels(ax, f"Bar Plot of {y} by {x}", str(x), str(y))

class SpanPlot(PlotStrategy):
    """A concrete class to implement the span plot strategy."""

    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the span plot of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param start: The start of the span.
        :type start: int
        :param end: The end of the span.
        :type end: int
        :param color: The color of the span. Default is 'blue'.
        :type color: str, optional
        :param alpha: The transparency of the span. Default is 0.5.
        :type alpha: float, optional
        :return: None
        """
        start = kwargs.get('start', 0)
        end = kwargs.get('end', 0)
        color = kwargs.get('color', 'blue')
        alpha = kwargs.get('alpha', 0.5)
        ax.axvspan(start, end, color=color, alpha=alpha)
        self.set_title_and_labels(ax, "Span Plot", "", "")