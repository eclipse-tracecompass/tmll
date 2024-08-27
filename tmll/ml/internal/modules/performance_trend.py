import pandas as pd

from tmll.tmll_client import TMLLClient
from tmll.ml.internal.modules.base_module import BaseModule

from tmll.ml.preprocess.normalizer import Normalizer

from tmll.common.models.output import Output

import matplotlib.pyplot as plt


TARGET_OUTPUTS = [
    Output.from_dict({
        "name": "CPU Usage",
        "id": "org.eclipse.tracecompass.analysis.os.linux.core.cpuusage.CpuUsageDataProvider",
        "type": "TREE_TIME_XY"
    }),
    Output.from_dict({
        "name": "Disk I/O View",
        "id": "org.eclipse.tracecompass.analysis.os.linux.core.inputoutput.DisksIODataProvider",
        "type": "TREE_TIME_XY"
    }),
    Output.from_dict({
        "name": "Memory Usage",
        "id": "org.eclipse.tracecompass.analysis.os.linux.core.kernelmemoryusage",
        "type": "TREE_TIME_XY"
    }),
]


class PerformanceTrend(BaseModule):

    def __init__(self, client: TMLLClient):
        super().__init__(client=client)

    def process(self):
        self.logger.info("Starting performance trend analysis...")

        self.logger.info("Fetching data...")
        data = self._fetch_data(outputs=TARGET_OUTPUTS[:])
        if not data:
            self.logger.error("No data fetched")
            return

        final_dataframe = pd.DataFrame()
        for output in TARGET_OUTPUTS[:]:
            if output.id not in data:
                self.logger.warning(f"The trace data does not contain the output {output.name}.")
                continue

            self.logger.info(f"Processing output {output.name}.")

            # Join the dataframes together on the time (i.e., x-axis) column
            if isinstance(data[output.id], pd.DataFrame):
                dataframe: pd.DataFrame = data[output.id] # type: ignore
                pass
            elif isinstance(data[output.id], dict):
                for key, value in data[output.id].items():
                    if isinstance(value, pd.DataFrame):
                        dataframe: pd.DataFrame = value

                        # Rename the y-axis column to the name of the output
                        dataframe = dataframe.rename(columns={'y': output.name, 'x': 'timestamp'})

                        # Join the dataframes together on the time (i.e., x-axis) column, which is 'x' in this case
                        if final_dataframe.empty:
                            final_dataframe = dataframe
                        else:
                            final_dataframe = pd.merge(final_dataframe, dataframe, on='timestamp', how='outer')
            else:
                self.logger.error(f"Unsupported data type for output {output.name}.")
                continue

        if not final_dataframe.empty:
            import numpy as np
            from matplotlib import colors as mcolors

            usage_metrics = ['CPU Usage', 'Memory Usage', 'Disk I/O View']

            # Check if the data contains the required metrics
            for metric_ in usage_metrics:
                if metric_ not in final_dataframe.columns:
                    # If the metric is not found, remove it from the list
                    usage_metrics.remove(metric_)

            if not usage_metrics:
                self.logger.warning("The data does not contain the required metrics.")
                return

            normalizer = Normalizer(final_dataframe, method='minmax')
            final_dataframe = normalizer.normalize(target_features=usage_metrics)
            time_ns = np.array(final_dataframe['timestamp'])
            time_sec = (time_ns - time_ns[0]) / 1e9

            time_interval = 0.01  # 100 ms in seconds
            num_cells = 1000  # 100 cells representing 100 ms within each time_interval

            # Determine the total duration
            total_duration = time_sec[-1]

            # Create a 3D array for the plot (metrics x cells x time points)
            num_time_points = int(np.ceil(total_duration / time_interval))
            heatmap_data = np.zeros((len(usage_metrics), num_cells, num_time_points))

            for metric_index, metric in enumerate(usage_metrics):
                usage_values = np.array(final_dataframe[metric])
                for i, t in enumerate(time_sec):
                    time_index = int(t / time_interval)
                    cell_index = int((t % time_interval) / (time_interval / num_cells))
                    
                    if time_index < num_time_points and cell_index < num_cells:
                        if heatmap_data[metric_index, cell_index, time_index] == 0:
                            heatmap_data[metric_index, cell_index, time_index] = usage_values[i]
                        else:
                            heatmap_data[metric_index, cell_index, time_index] = (heatmap_data[metric_index, cell_index, time_index] + usage_values[i]) / 2

            fig, axes = plt.subplots(len(usage_metrics), 1, figsize=(12, 5*len(usage_metrics)), sharex=True)
            fig.suptitle('Stacked Heatmap: CPU, Memory, and Disk Usage Over Time', fontsize=16)

            for i, (ax, metric) in enumerate(zip(axes, usage_metrics)):
                norm = mcolors.TwoSlopeNorm(vmin=np.min(heatmap_data[i]), 
                                            vcenter=np.mean(heatmap_data[i]), 
                                            vmax=np.max(heatmap_data[i]))
                im = ax.imshow(heatmap_data[i], aspect='auto', cmap='gray_r', norm=norm)
                ax.set_ylabel(f'{metric}\nMilliseconds\nwithin each 10ms')
                fig.colorbar(im, ax=ax, label=metric)
                
                if i == len(usage_metrics) - 1:  # Only set xlabel for the bottom subplot
                    ax.set_xlabel('Time (s)')
                
                ax.grid(True)

            plt.tight_layout()
            plt.show()
