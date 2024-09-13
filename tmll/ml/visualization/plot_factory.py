from tmll.ml.visualization.plot_types import *


class PlotFactory:
    """
    Based on the "Factory Method" design pattern, this module provides a static method to 
    create a plot strategy based on the plot type.
    """

    @staticmethod
    def create_plot(plot_type: str) -> PlotStrategy:
        """
        Create a plot strategy based on the plot type.

        :param plot_type: The type of plot to create
        :type plot_type: str
        :return: The plot strategy based on the plot type
        :rtype: PlotStrategy
        """

        plot_types = {
            'time_series': TimeSeriesPlot(),
            'scatter': ScatterPlot(),
            'histogram': HistogramPlot(),
            'box': BoxPlot(),
            'violin': ViolinPlot(),
            'heatmap': HeatmapPlot(),
            'pair': PairPlot(),
            'bar': BarPlot()
        }

        return plot_types.get(plot_type, TimeSeriesPlot())