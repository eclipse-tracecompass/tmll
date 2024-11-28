from typing import Optional
import matplotlib.pyplot as plt

class PlotUtils():
    """
    Utility class for plotting.
    """

    @staticmethod
    def set_standard_legend_style(ax, handles, labels, padding_factor: float = 0.02, title: Optional[str] = None) -> None:
        """
        Helper method to standardize legend appearance across all plots with adaptive positioning.
        
        :param ax: Matplotlib axes object
        :type ax: plt.Axes
        :param padding_factor: Padding factor for legend position adjustment
        :type padding_factor: float
        :param title: Title for the legend
        :type title: Optional[str]
        """
        # Get the figure and axes dimensions
        fig = ax.get_figure()
        fig_width_inches = fig.get_figwidth()
        
        # Get the axes position in figure coordinates
        bbox = ax.get_position()
        plot_width = bbox.width
        
        # Calculate the right position for legend
        # Move legend outside plot area with some padding
        legend_x = 1 + (padding_factor * plot_width)
        
        legend = ax.legend(
            handles,
            labels,
            bbox_to_anchor=(legend_x, 1),
            title=title,
            fontsize=12,
            title_fontsize=12,
            frameon=True,
            borderaxespad=0,
            loc='upper left'
        )
        
        # Remove the default title padding
        legend._legend_box.align = "left"
        
        # Get the legend width in figure coordinates
        legend_width = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).width
        
        # If legend goes beyond figure bounds, adjust figure size to accommodate it
        total_width_needed = bbox.x1 + (legend_width * 1.1)  # 1.1 adds a small right margin
        if total_width_needed > 1:
            # Calculate new figure width
            new_fig_width = fig_width_inches / bbox.x1
            fig.set_figwidth(new_fig_width)
            
            # Update tight_layout with new dimensions
            plt.tight_layout()
            
            # Reposition legend after tight_layout adjustment
            bbox = ax.get_position()
            legend_x = 1 + (padding_factor * bbox.width)
            legend.set_bbox_to_anchor((legend_x, 1))