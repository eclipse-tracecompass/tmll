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

        Args:
            plot_type (str): The type of plot to create.

        Returns:
            PlotStrategy: The plot strategy based on the plot type.
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