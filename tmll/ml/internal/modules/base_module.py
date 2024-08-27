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