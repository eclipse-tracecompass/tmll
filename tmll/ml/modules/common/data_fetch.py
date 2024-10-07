import pandas as pd
from typing import List, Dict, Union

from tmll.tmll_client import TMLLClient
from tmll.common.models.output import Output

class DataFetcher:
    def __init__(self, client: TMLLClient):
        self.client = client
        self.logger = client.logger

    def fetch_data(self, outputs: List[Output], force_reload: bool = False) -> Union[None, Dict[str, pd.DataFrame]]:
        """
        Fetch and process data for the given outputs.

        :param outputs: List of Output objects to fetch data for
        :type outputs: List[Output]
        :param force_reload: Whether to force reload the data
        :type force_reload: bool
        :return: The processed data for the given outputs
        :rtype: Union[None, Dict[str, pd.DataFrame]]
        """
        desired_outputs = outputs
        if not desired_outputs:
            desired_outputs = [o["output"] for o in self.client.outputs]

        self.logger.info("Fetching data...")
        data = self._fetch_data(outputs=desired_outputs, force_reload=force_reload)
        if not data:
            self.logger.error("No data fetched")
            return None

        dataframes = {}
        for output in desired_outputs:
            if output.id not in data:
                self.logger.warning(f"The trace data does not contain the output {output.name}.")
                continue

            self.logger.info(f"Processing output {output.name}.")

            if isinstance(data[output.id], dict):
                for key, value in data[output.id].items():
                    if isinstance(value, pd.DataFrame):
                        dataframe = value
                        dataframe = dataframe.rename(columns={'y': output.name, 'x': 'timestamp'})
                        if dataframes.get(f"{output.id}${key}", None) is None:
                            dataframes[f"{output.id}${key}"] = dataframe
                        else:
                            dataframes[f"{output.id}${key}"] = pd.merge(dataframes[f"{output.id}_{key}"], dataframe, on='timestamp', how='outer')
            elif isinstance(data[output.id], pd.DataFrame):
                dataframes[output.id] = data[output.id]

        return dataframes
    
    def _fetch_data(self, outputs: List[Output], force_reload: bool = False) -> Union[None, Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]:
        """
        Fetch the data from the given outputs.

        :param outputs: The outputs to fetch the data from
        :type outputs: List[Output]
        :param force_reload: Whether to force reload the data
        :type force_reload: bool
        :return: The fetched data
        :rtype: Union[None, Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]
        """
        custom_output_ids = [output.id for output in outputs]
        self.client.fetch_outputs(custom_output_ids=custom_output_ids, force_reload=force_reload)
        return self.client.fetch_data(custom_output_ids=custom_output_ids, separate_columns=True)