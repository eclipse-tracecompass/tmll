from typing import Dict, List, Union
import pandas as pd

from tmll.common.models.output import Output

from tmll.common.services.logger import Logger
from tmll.tmll_client import TMLLClient


class BaseModule:

    def __init__(self, client: TMLLClient):
        self.client = client
        
        self.logger = Logger(self.__class__.__name__)

    def _fetch_data(self, outputs: List[Output]) -> Union[None, Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]:
        self.client.fetch_outputs(custom_output_ids=[output.id for output in outputs])
        return self.client.fetch_data(custom_output_ids=[output.id for output in outputs], separate_columns=True)
    
    def _process(self, target_outputs: List[Output]) -> None:
        self.logger.info("Fetching data...")
        data = self._fetch_data(outputs=target_outputs)
        if not data:
            self.logger.error("No data fetched")
            return

        dataframes: Dict[str, pd.DataFrame] = {}
        for output in target_outputs:
            if output.id not in data:
                self.logger.warning(f"The trace data does not contain the output {output.name}.")
                continue

            self.logger.info(f"Processing output {output.name}.")

            if isinstance(data[output.id], pd.DataFrame):
                dataframe: pd.DataFrame = data[output.id] # type: ignore
                dataframes[output.id] = dataframe
            elif isinstance(data[output.id], dict):
                for _, df in data[output.id].items():
                    if isinstance(df, pd.DataFrame):
                        dataframe: pd.DataFrame = df
                        dataframes[output.id] = dataframe
            else:
                self.logger.error(f"Unsupported data type for output {output.name}.")
                continue
    
    # Abstract method for processing the data
    def process(self):
        self.logger.warning("The process method must be implemented in the child class.")