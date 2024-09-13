import pandas as pd

from tmll.tmll_client import TMLLClient
from tmll.ml.modules.base_module import BaseModule

from tmll.ml.preprocess.normalizer import Normalizer
from tmll.ml.preprocess.outlier_remover import OutlierRemover

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
# Seasonal analysis -> CPU Usage/ Network / File Access
# Resources  View -> Frequency analysis for power consumption (Resources Status Data Provider, Control Flow View)

class PerformanceTrend(BaseModule):

    def __init__(self, client: TMLLClient):
        super().__init__(client=client)

    def process(self) -> None:
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
            from matplotlib.dates import DateFormatter

            # Convert timestamp to datetime
            final_dataframe['timestamp'] = pd.to_datetime(final_dataframe['timestamp'], unit='ns')

            # Set the timestamp as the index
            final_dataframe.set_index('timestamp', inplace=True)

            # Resample the data to 1ms intervals and fill missing values with 0
            final_dataframe = final_dataframe.resample('1ms').mean().fillna(0)

            # Remove outliers
            outlier_remover = OutlierRemover(dataset=final_dataframe)
            final_dataframe = outlier_remover.remove_outliers()

            # Normalize the data (except for the timestamp column)
            normalizer = Normalizer(dataset=final_dataframe, method='minmax')
            final_dataframe = normalizer.normalize()

            # Combined plot for all metrics
            _, ax = plt.subplots(figsize=(15, 3))
            for column in final_dataframe.columns:
                ax.plot(final_dataframe.index, final_dataframe[column], label=column)

            ax.set_title('Combined System Metrics Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Normalized Usage')
            ax.legend()

            # Format x-axis to show readable dates
            ax.xaxis.set_major_formatter(DateFormatter('%S.%f'))
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()
