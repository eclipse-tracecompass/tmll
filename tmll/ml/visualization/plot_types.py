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
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :param marker: The marker style for the plot. Default is None.
        :type marker: str, optional
        :param linewidth: The width of the line. Default is 1.
        :type linewidth: int, optional
        :return: None
        """
        label = kwargs.get('label', None)
        x = kwargs.get('x', 'timestamp')
        y = kwargs.get('y')
        hue = kwargs.get('hue', None)
        color = kwargs.get('color', 'blue')
        alpha = kwargs.get('alpha', 1.0)
        is_top = kwargs.get('is_top', False)
        marker = kwargs.get('marker', None)
        linewidth = kwargs.get('linewidth', 1)
        
        zorder = 0
        if is_top:
            zorder = max([line.get_zorder() for line in ax.lines]) + 1
        
        sns.lineplot(data=data, x=x, y=y, hue=hue, ax=ax, color=color, alpha=alpha, label=label, marker=marker, zorder=zorder, linewidth=linewidth)
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
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :param marker: The marker style for the plot. Default is '*'.
        :type marker: str, optional
        :param s: The size of the marker. Default is 1.
        :type s: int, optional
        :return: None
        """
        x = kwargs.get('x')
        y = kwargs.get('y')
        hue = kwargs.get('hue', None)
        color = kwargs.get('color', 'blue')
        label = kwargs.get('label', None)
        alpha = kwargs.get('alpha', 1.0)
        is_top = kwargs.get('is_top', False)
        marker = kwargs.get('marker', '*')
        s = kwargs.get('s', 1)
        
        zorder = 0
        if is_top:
            zorder = max([scatter.get_zorder() for scatter in ax.collections]) + 1
        
        sns.scatterplot(data=data, x=x, y=y, ax=ax, color=color, hue=hue, label=label, alpha=alpha, marker=marker, s=s, zorder=zorder)
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
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :return: None
        """
        column = kwargs.get('column')
        color = kwargs.get('color', 'blue')
        bins = kwargs.get('bins', 50)
        label = kwargs.get('label', None)
        alpha = kwargs.get('alpha', 1.0)
        is_top = kwargs.get('is_top', False)
        
        if isinstance(bins, dict):
            bins = bins.get('bins', 50)
        
        zorder = 0
        if is_top:
            zorder = max([patch.get_zorder() for patch in ax.patches]) + 1
        
        sns.histplot(data=data, x=column, bins=bins, ax=ax, kde=True, color=color, label=label, alpha=alpha, zorder=zorder)
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
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :return: None
        """
        x = kwargs.get('x')
        y = kwargs.get('y')
        color = kwargs.get('color', 'blue')
        label = kwargs.get('label', None)
        alpha = kwargs.get('alpha', 1.0)
        is_top = kwargs.get('is_top', False)
        
        zorder = 0
        if is_top:
            zorder = max([patch.get_zorder() for patch in ax.patches]) + 1
        
        sns.boxplot(data=data, x=x, y=y, ax=ax, color=color, label=label, alpha=alpha, zorder=zorder)
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
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :return: None
        """
        x = kwargs.get('x')
        y = kwargs.get('y')
        color = kwargs.get('color', 'blue')
        label = kwargs.get('label', None)
        alpha = kwargs.get('alpha', 1.0)
        is_top = kwargs.get('is_top', False)
        
        zorder = 0
        if is_top:
            zorder = max([collection.get_zorder() for collection in ax.collections]) + 1
        
        sns.violinplot(data=data, x=x, y=y, ax=ax, color=color, label=label, alpha=alpha, zorder=zorder)
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
        :param mask: The mask for the heatmap. Default is None.
        :type mask: Any, optional
        :param cmap: The name of the colormap. Default is 'viridis'.
        :type cmap: str, optional
        :param annot: Whether to annotate the heatmap. Default is True.
        :type annot: bool, optional
        :param fmt: The format of the annotation. Default is '.2f'.
        :type fmt: str, optional
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :return: None
        """
        mask = kwargs.get('mask', None)
        cmap = kwargs.get('cmap', 'viridis')
        annot = kwargs.get('annot', True)
        fmt = kwargs.get('fmt', '.2f')
        label = kwargs.get('label', None)
        alpha = kwargs.get('alpha', 1.0)
        is_top = kwargs.get('is_top', False)
        
        zorder = 0
        if is_top:
            zorder = max([im.get_zorder() for im in ax.images]) + 1
        
        sns.heatmap(data, ax=ax, cmap=cmap, annot=annot, fmt=fmt, alpha=alpha, zorder=zorder, mask=mask)
        if label:
            ax.collections[0].set_label(label)
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
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :return: None
        """
        vars = kwargs.get('vars')
        hue = kwargs.get('hue')
        label = kwargs.get('label', None)
        alpha = kwargs.get('alpha', 1.0)
        is_top = kwargs.get('is_top', False)
        
        # Note: is_top is not applicable for pairplot as it creates its own figure
        sns.pairplot(data=data, vars=vars, hue=str(hue), plot_kws={'alpha': alpha})
        if label:
            plt.suptitle(label, y=1.02)
        else:
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
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :return: None
        """
        x = kwargs.get('x')
        y = kwargs.get('y')
        hue = kwargs.get('hue', None)
        color = kwargs.get('color', 'blue')
        label = kwargs.get('label', None)
        alpha = kwargs.get('alpha', 1.0)
        is_top = kwargs.get('is_top', False)
        
        zorder = 0
        if is_top:
            zorder = max([patch.get_zorder() for patch in ax.patches]) + 1
        
        sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, color=color, label=label, alpha=alpha, zorder=zorder)
        self.set_title_and_labels(ax, f"Bar Plot of {y} by {x}", str(x), str(y))

class SpanPlot(PlotStrategy):
    """A concrete class to implement the span plot strategy."""

    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the span plot of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot (not used in SpanPlot, but kept for consistency).
        :type data: Any
        :param start: The start of the span.
        :type start: float
        :param end: The end of the span.
        :type end: float
        :param color: The color of the span. Default is 'blue'.
        :type color: str, optional
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the span. Default is 0.5.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :return: None
        """
        start = kwargs.get('start', 0)
        end = kwargs.get('end', 0)
        color = kwargs.get('color', 'blue')
        label = kwargs.get('label', None)
        alpha = kwargs.get('alpha', 0.5)
        is_top = kwargs.get('is_top', False)

        zorder = 0
        if is_top:
            zorder = max([collection.get_zorder() for collection in ax.collections]) + 1

        ax.axvspan(start, end, color=color, alpha=alpha, zorder=zorder, label=label)
        self.set_title_and_labels(ax, "Span Plot", "", "")

class VLinePlot(PlotStrategy):
    """A concrete class to implement the vertical line plot strategy."""

    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the vertical line on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot (not used in VLinePlot, but kept for consistency).
        :type data: Any
        :param x: The x-coordinate of the vertical line.
        :type x: float
        :param color: The color of the line. Default is 'blue'.
        :type color: str, optional
        :param linestyle: The line style of the line. Default is '--'.
        :type linestyle: str, optional
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the line. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :return: None
        """
        x = kwargs.get('x', 0)
        color = kwargs.get('color', 'blue')
        linestyle = kwargs.get('linestyle', '--')
        label = kwargs.get('label', None)
        alpha = kwargs.get('alpha', 1.0)
        is_top = kwargs.get('is_top', False)

        zorder = 0
        if is_top:
            zorder = max([line.get_zorder() for line in ax.lines]) + 1

        ax.axvline(x=x, color=color, alpha=alpha, zorder=zorder, label=label, linestyle=linestyle)
        self.set_title_and_labels(ax, "Vertical Line Plot", "", "")

class HLinePlot(PlotStrategy):
    """A concrete class to implement the horizontal line plot strategy."""

    def plot(self, ax: Axes, data: Any, **kwargs) -> None:
        """
        Plot the horizontal line on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot (not used in HLinePlot, but kept for consistency).
        :type data: Any
        :param y: The y-coordinate of the horizontal line.
        :type y: float
        :param color: The color of the line. Default is 'blue'.
        :type color: str, optional
        :param linestyle: The line style of the line. Default is '--'.
        :type linestyle: str, optional
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the line. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :return: None
        """
        y = kwargs.get('y', 0)
        color = kwargs.get('color', 'blue')
        linestyle = kwargs.get('linestyle', '--')
        label = kwargs.get('label', None)
        alpha = kwargs.get('alpha', 1.0)
        is_top = kwargs.get('is_top', False)

        zorder = 0
        if is_top:
            zorder = max([line.get_zorder() for line in ax.lines]) + 1

        ax.axhline(y=y, color=color, alpha=alpha, zorder=zorder, label=label, linestyle=linestyle)
        self.set_title_and_labels(ax, "Horizontal Line Plot", "", "")