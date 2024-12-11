from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from tmll.ml.visualization.plot_strategy import PlotStrategy


class TimeSeriesPlot(PlotStrategy):
    """A concrete class to implement the time series plot strategy."""

    def plot(self, ax: Axes, data: Optional[Any] = None, **kwargs) -> None:
        """
        Plot the time series data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any, optional
        :param x: The name of the x-axis column.
        :type x: str, optional
        :param y: The name of the y-axis column.
        :type y: str, optional
        :param ax_title: The title of the plot. Default is "Time Series Plot".
        :type ax_title: str, optional
        :param x_label: The label for the x-axis. Default is "x-axis".
        :type x_label: str, optional
        :param y_label: The label for the y-axis. Default is "y-axis".
        :type y_label: str, optional
        :param color: The color of the plot. Default is "blue".
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
        """
        x, y = self._get_x_y(data, **kwargs)

        ax_title = kwargs.get("ax_title", "Time Series Plot")
        x_label = kwargs.get("x_label", "x-axis")
        y_label = kwargs.get("y_label", "y-axis")
        color = kwargs.get("color", "blue")
        label = kwargs.get("label", None)
        alpha = kwargs.get("alpha", 1.0)
        is_top = kwargs.get("is_top", False)
        marker = kwargs.get("marker", None)
        linewidth = kwargs.get("linewidth", 1)

        zorder = 0
        if is_top:
            zorder = max([line.get_zorder() for line in ax.lines]) + 1

        ax.plot(x, y, color=color, label=label, alpha=alpha, zorder=zorder, marker=marker, linewidth=linewidth)
        self.set_title_and_labels(ax, ax_title, x_label, y_label)


class ScatterPlot(PlotStrategy):
    """A concrete class to implement the scatter plot strategy."""

    def plot(self, ax: Axes, data: Optional[Any] = None, **kwargs) -> None:
        """
        Plot the scatter plot of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any, optional
        :param x: The name of the x-axis column.
        :type x: str, optional
        :param y: The name of the y-axis column.
        :type y: str, optional
        :param ax_title: The title of the plot. Default is "Scatter Plot".
        :type ax_title: str, optional
        :param x_label: The label for the x-axis. Default is "x-axis".
        :type x_label: str, optional
        :param y_label: The label for the y-axis. Default is "y-axis".
        :type y_label: str, optional
        :param color: The color of the plot. Default is "blue".
        :type color: str, optional
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        :param marker: The marker style for the plot. Default is "*".
        :type marker: str, optional
        :param s: The size of the marker. Default is 1.
        :type s: int, optional
        """
        x, y = self._get_x_y(data, **kwargs)

        ax_title = kwargs.get("ax_title", "Scatter Plot")
        x_label = kwargs.get("x_label", "x-axis")
        y_label = kwargs.get("y_label", "y-axis")
        color = kwargs.get("color", "blue")
        label = kwargs.get("label", None)
        alpha = kwargs.get("alpha", 1.0)
        is_top = kwargs.get("is_top", False)
        marker = kwargs.get("marker", "*")
        s = kwargs.get("s", 1)

        zorder = 0
        if is_top:
            zorder = max([scatter.get_zorder() for scatter in ax.collections]) + 1

        ax.scatter(x, y, color=color, label=label, alpha=alpha, zorder=zorder, marker=marker, s=s)
        self.set_title_and_labels(ax, ax_title, x_label, y_label)


class HistogramPlot(PlotStrategy):
    """A concrete class to implement the histogram plot strategy."""

    def plot(self, ax: Axes, data: Optional[Any] = None, **kwargs) -> None:
        """
        Plot the histogram of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any, optional
        :param column: The name of the column.
        :type column: str, optional
        :param ax_title: The title of the plot. Default is "Histogram Plot".
        :type ax_title: str, optional
        :param x_label: The label for the x-axis. Default is "x-axis".
        :type x_label: str, optional
        :param y_label: The label for the y-axis. Default is "Frequency".
        :type y_label: str, optional
        :param color: The color of the plot. Default is "blue".
        :type color: str, optional
        :param bins: The number of bins for the histogram. Default is 50.
        :type bins: int, optional
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        """
        if data is None:
            raise ValueError("Data is required for plotting")

        ax_title = kwargs.get("ax_title", "Histogram Plot")
        x_label = kwargs.get("x_label", "x-axis")
        y_label = kwargs.get("y_label", "Frequency")
        column = kwargs.get("column", None)
        color = kwargs.get("color", "blue")
        bins = kwargs.get("bins", 50)
        label = kwargs.get("label", None)
        alpha = kwargs.get("alpha", 1.0)
        is_top = kwargs.get("is_top", False)

        if isinstance(bins, dict):
            bins = bins.get("bins", 50)

        zorder = 0
        if is_top:
            zorder = max([patch.get_zorder() for patch in ax.patches]) + 1

        ax.hist(data, color=color, bins=bins, label=label, alpha=alpha, zorder=zorder)
        self.set_title_and_labels(ax, ax_title, x_label, y_label)


class BoxPlot(PlotStrategy):
    """A concrete class to implement the box plot strategy."""

    def plot(self, ax: Axes, data: Optional[Any] = None, **kwargs) -> None:
        """
        Plot the box plot of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any, optional
        :param x: The name of the x-axis column.
        :type x: str, optional
        :param y: The name of the y-axis column.
        :type y: str, optional
        :param ax_title: The title of the plot. Default is "Box Plot".
        :type ax_title: str, optional
        :param x_label: The label for the x-axis. Default is "x-axis".
        :type x_label: str, optional
        :param y_label: The label for the y-axis. Default is "y-axis".
        :type y_label: str, optional
        :param color: The color of the plot. Default is "blue".
        :type color: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        """
        x, y = self._get_x_y(data, **kwargs)

        ax_title = kwargs.get("ax_title", "Box Plot")
        x_label = kwargs.get("x_label", "x-axis")
        y_label = kwargs.get("y_label", "y-axis")
        color = kwargs.get("color", "blue")
        alpha = kwargs.get("alpha", 1.0)
        is_top = kwargs.get("is_top", False)

        zorder = 0
        if is_top:
            zorder = max([patch.get_zorder() for patch in ax.patches]) + 1

        ax.boxplot(y, positions=x, patch_artist=True, boxprops=dict(facecolor=color, alpha=alpha), zorder=zorder)
        self.set_title_and_labels(ax, ax_title, x_label, y_label)


class HeatmapPlot(PlotStrategy):
    """A concrete class to implement the heatmap plot strategy."""

    def plot(self, ax: Axes, data: Optional[Any] = None, **kwargs) -> None:
        """
        Plot the heatmap of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any, optional
        :param ax_title: The title of the plot. Default is "Heatmap Plot".
        :type ax_title: str, optional
        :param mask: The mask for the heatmap. Default is None.
        :type mask: Any, optional
        :param cmap: The colormap for the heatmap. Default is "viridis".
        :type cmap: str, optional
        :param annot: Whether to annotate the heatmap. Default is True.
        :type annot: bool, optional
        :param fmt: The format for the annotation. Default is ".2f".
        :type fmt: str, optional
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        """
        if data is None:
            raise ValueError("Data is required for plotting")

        ax_title = kwargs.get("ax_title", "Heatmap Plot")
        mask = kwargs.get("mask", None)
        cmap = kwargs.get("cmap", "viridis")
        annot = kwargs.get("annot", True)
        fmt = kwargs.get("fmt", ".2f")
        label = kwargs.get("label", None)
        alpha = kwargs.get("alpha", 1.0)
        is_top = kwargs.get("is_top", False)

        if mask is not None:
            masked_data = np.ma.masked_array(data, mask=mask)
        else:
            masked_data = data

        zorder = 0
        if is_top:
            zorder = max([im.get_zorder() for im in ax.images]) + 1

        im = ax.imshow(masked_data, cmap=cmap, aspect="auto", alpha=alpha, zorder=zorder)
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.index)

        if annot:
            for i in range(masked_data.shape[0]):
                for j in range(masked_data.shape[1]):
                    if mask is None or not mask[i, j]:
                        value = masked_data[i, j]
                        text = f"{value:{fmt}}"
                        color_value = plt.get_cmap(cmap)(im.norm(value))
                        brightness = np.mean(color_value[:3])
                        text_color = 'white' if brightness < 0.5 else 'black'
                        ax.text(j, i, text, ha='center', va='center', color=text_color)

        if label:
            im.set_label(label)

        self.set_title_and_labels(ax, ax_title, "", "")


class BarPlot(PlotStrategy):
    """A concrete class to implement the bar plot strategy."""

    def plot(self, ax: Axes, data: Optional[Any] = None, **kwargs) -> None:
        """
        Plot the bar plot of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot.
        :type data: Any, optional
        :param x: The name of the x-axis column.
        :type x: str, optional
        :param y: The name of the y-axis column.
        :type y: str, optional
        :param ax_title: The title of the plot. Default is "Bar Plot".
        :type ax_title: str, optional
        :param x_label: The label for the x-axis. Default is "x-axis".
        :type x_label: str, optional
        :param y_label: The label for the y-axis. Default is "y-axis".
        :type y_label: str, optional
        :param color: The color of the plot. Default is "blue".
        :type color: str, optional
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the plot. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        """
        x, y = self._get_x_y(data, **kwargs)

        ax_title = kwargs.get("ax_title", "Bar Plot")
        x_label = kwargs.get("x_label", "x-axis")
        y_label = kwargs.get("y_label", "y-axis")
        color = kwargs.get("color", "blue")
        label = kwargs.get("label", None)
        alpha = kwargs.get("alpha", 1.0)
        is_top = kwargs.get("is_top", False)

        zorder = 0
        if is_top:
            zorder = max([patch.get_zorder() for patch in ax.patches]) + 1

        ax.bar(x, y, color=color, label=label, alpha=alpha, zorder=zorder)
        self.set_title_and_labels(ax, ax_title, x_label, y_label)


class SpanPlot(PlotStrategy):
    """A concrete class to implement the span plot strategy."""

    def plot(self, ax: Axes, data: Optional[Any] = None, **kwargs) -> None:
        """
        Plot the span plot of the data on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot (not used in SpanPlot, but kept for consistency).
        :type data: Any, optional
        :param start: The start of the span.
        :type start: float
        :param end: The end of the span.
        :type end: float
        :param ax_title: The title of the plot. Default is "Span Plot".
        :type ax_title: str, optional
        :param color: The color of the span. Default is "blue".
        :type color: str, optional
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the span. Default is 0.5.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        """
        start = kwargs.get("start", 0)
        end = kwargs.get("end", 0)

        ax_title = kwargs.get("ax_title", "Span Plot")
        color = kwargs.get("color", "blue")
        label = kwargs.get("label", None)
        alpha = kwargs.get("alpha", 0.5)
        is_top = kwargs.get("is_top", False)

        zorder = 0
        if is_top:
            zorder = max([collection.get_zorder() for collection in ax.collections]) + 1

        ax.axvspan(start, end, color=color, alpha=alpha, zorder=zorder, label=label)
        self.set_title_and_labels(ax, ax_title, "", "")


class VLinePlot(PlotStrategy):
    """A concrete class to implement the vertical line plot strategy."""

    def plot(self, ax: Axes, data: Optional[Any] = None, **kwargs) -> None:
        """
        Plot the vertical line on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot (not used in VLinePlot, but kept for consistency).
        :type data: Any, optional
        :param x: The x-coordinate of the vertical line.
        :type x: float
        :param ax_title: The title of the plot. Default is "Vertical Line Plot".
        :type ax_title: str, optional
        :param color: The color of the line. Default is "blue".
        :type color: str, optional
        :param linestyle: The line style of the line. Default is "--".
        :type linestyle: str, optional
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the line. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        """
        x = kwargs.get("x", 0)

        ax_title = kwargs.get("ax_title", "Vertical Line Plot")
        color = kwargs.get("color", "blue")
        linestyle = kwargs.get("linestyle", "--")
        label = kwargs.get("label", None)
        alpha = kwargs.get("alpha", 1.0)
        is_top = kwargs.get("is_top", False)

        zorder = 0
        if is_top:
            zorder = max([line.get_zorder() for line in ax.lines]) + 1

        ax.axvline(x=x, color=color, alpha=alpha, zorder=zorder, label=label, linestyle=linestyle)
        self.set_title_and_labels(ax, ax_title, "", "")


class HLinePlot(PlotStrategy):
    """A concrete class to implement the horizontal line plot strategy."""

    def plot(self, ax: Axes, data: Optional[Any] = None, **kwargs) -> None:
        """
        Plot the horizontal line on the given axes.

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot (not used in HLinePlot, but kept for consistency).
        :type data: Any, optional
        :param y: The y-coordinate of the horizontal line.
        :type y: float
        :param ax_title: The title of the plot. Default is "Horizontal Line Plot".
        :type ax_title: str, optional
        :param color: The color of the line. Default is "blue".
        :type color: str, optional
        :param linestyle: The line style of the line. Default is "--".
        :type linestyle: str, optional
        :param label: The label for the plot legend. Default is None.
        :type label: str, optional
        :param alpha: The transparency of the line. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        """
        y = kwargs.get("y", 0)

        ax_title = kwargs.get("ax_title", "Horizontal Line Plot")
        color = kwargs.get("color", "blue")
        linestyle = kwargs.get("linestyle", "--")
        label = kwargs.get("label", None)
        alpha = kwargs.get("alpha", 1.0)
        is_top = kwargs.get("is_top", False)

        zorder = 0
        if is_top:
            zorder = max([line.get_zorder() for line in ax.lines]) + 1

        ax.axhline(y=y, color=color, alpha=alpha, zorder=zorder, label=label, linestyle=linestyle)
        self.set_title_and_labels(ax, ax_title, "", "")


class AnnotatePlot(PlotStrategy):
    """A concrete class to implement the annotation plot strategy."""

    def plot(self, ax: Axes, data: Optional[Any] = None, **kwargs) -> None:
        """
        Plot the annotation on the given axes (ax.annotate).

        :param ax: The axes to plot the data.
        :type ax: Axes
        :param data: The data to plot (not used in AnnotatePlot, but kept for consistency).
        :type data: Any, optional
        :param text: The text to annotate. Default is "Annotation".
        :type text: str, optional
        :param xy: The position to annotate. Default is (0, 0).
        :type xy: tuple, optional
        :param xytext: The offset position of the annotation. Default is (5, 0).
        :type xytext: tuple, optional
        :param textcoords: The coordinate system for the text. Default is "offset points".
        :type textcoords: str, optional
        :param rotation: The rotation of the text. Default is 0.
        :type rotation: float, optional
        :param verticalalignment: The vertical alignment of the text. Default is "bottom".
        :type verticalalignment: str, optional
        :param ax_title: The title of the plot. Default is "Annotation Plot".
        :type ax_title: str, optional
        :param color: The color of the text. Default is "black".
        :type color: str, optional
        :param alpha: The transparency of the text. Default is 1.0.
        :type alpha: float, optional
        :param is_top: Whether to place this plot on top of others. Default is False.
        :type is_top: bool, optional
        """
        text = kwargs.get("text", "Annotation")
        xy = (kwargs.get("xy", 0), ax.get_ylim()[1] * 0.9)
        xytext = kwargs.get("xytext", (5, 0))
        textcoords = kwargs.get("textcoords", "offset points")
        rotation = kwargs.get("rotation", 0)
        verticalalignment = kwargs.get("verticalalignment", "bottom")

        ax_title = kwargs.get("ax_title", "Annotation Plot")
        color = kwargs.get("color", "black")
        alpha = kwargs.get("alpha", 1.0)
        is_top = kwargs.get("is_top", False)

        zorder = 0
        if is_top:
            zorder = max([text.get_zorder() for text in ax.texts]) + 1

        ax.annotate(text, xy=xy, xytext=xytext, textcoords=textcoords, rotation=rotation,
                    verticalalignment=verticalalignment, color=color, alpha=alpha, zorder=zorder)
        self.set_title_and_labels(ax, ax_title, "", "")
