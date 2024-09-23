import pandas as pd
from typing import List, Dict, Union

from tmll.tmll_client import TMLLClient
from tmll.common.models.output import Output

class DataFetcher:
    def __init__(self, client: TMLLClient):
        self.client = client
        self.logger = client.logger

    def fetch_data(self, outputs: List[Output]) -> Union[None, pd.DataFrame]:
        """
        Fetch and process data for the given outputs.

        :param outputs: List of Output objects to fetch data for
        :type outputs: List[Output]
        :return: Dictionary of processed DataFrames
        :rtype: Dict[str, pd.DataFrame]
        """
        self.logger.info("Fetching data...")
        data = self._fetch_data(outputs)
        if not data:
            self.logger.error("No data fetched")
            return None

        final_dataframe = pd.DataFrame()
        for output in outputs:
            if output.id not in data:
                self.logger.warning(f"The trace data does not contain the output {output.name}.")
                continue

            self.logger.info(f"Processing output {output.name}.")

            if isinstance(data[output.id], dict):
                for _, value in data[output.id].items():
                    if isinstance(value, pd.DataFrame):
                        dataframe: pd.DataFrame = value
                        dataframe = dataframe.rename(columns={'y': output.name, 'x': 'timestamp'})
                        if final_dataframe.empty:
                            final_dataframe = dataframe
                        else:
                            final_dataframe = pd.merge(final_dataframe, dataframe, on='timestamp', how='outer')

        final_dataframe = final_dataframe.fillna(0)
        return final_dataframe
    
    def _fetch_data(self, outputs: List[Output]) -> Union[None, Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]:
        """
        Fetch the data from the given outputs.

        :param outputs: The outputs to fetch the data from
        :type outputs: List[Output]
        :return: The fetched data
        :rtype: Union[None, Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]
        """
        
        self.client.fetch_outputs(custom_output_ids=[output.id for output in outputs])
        return self.client.fetch_data(custom_output_ids=[output.id for output in outputs], separate_columns=True)