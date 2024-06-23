"""
This module contains the TSPClient class, which is used to interact with the TSP server.


"""


from typing import List, Dict, Optional, Union, Tuple, Any

from tmll.tsp.models.response.base import BaseResponse
from tmll.tsp.models.response.experiment import ExperimentResponse
from tmll.tsp.models.response.trace import TraceResponse

from tmll.tsp.services.trace_service import TraceService
from tmll.tsp.services.experiment_service import ExperimentService
from tmll.tsp.services.output_service import OutputService
from tmll.tsp.services.tree_service import TreeService
from tmll.tsp.services.data_service import DataService


class TSPClient():

    def __init__(self, tsp_client_path: str, tsp_client_name: str) -> None:
        self.tsp_client_path = tsp_client_path
        self.tsp_client_name = tsp_client_name

        self.trace_service = TraceService(tsp_client_path, tsp_client_name)
        self.experiment_service = ExperimentService(tsp_client_path, tsp_client_name)
        self.output_service = OutputService(tsp_client_path, tsp_client_name)
        self.tree_service = TreeService(tsp_client_path, tsp_client_name)
        self.data_service = DataService(tsp_client_path, tsp_client_name)

    def import_traces(self, traces: Dict[str, str], create_experiment: bool = False,
                      experiment_name: str = "") -> Union[BaseResponse[List[TraceResponse]], Tuple[BaseResponse[List[TraceResponse]], BaseResponse[ExperimentResponse]]]:
        """Import traces into the TSP server.

        Args:
            traces (Dict[str, str]): A dictionary containing the traces' names and contents.
            create_experiment (bool, optional): If True, it will create an experiment with the imported traces. Default to False.
            experiment_name (str, optional): The name of the experiment to create. Defaults to "".

        Returns:
            BaseResponse[List[TraceResponse]]: A response containing the traces' information.
        """

        # Import traces into the TSP server
        trace_res = self.trace_service.add_traces(traces)

        # If there was an error or no traces were created, return the trace_res
        if trace_res.error or not trace_res.result:
            return trace_res

        if create_experiment:
            trace_uuids = [trace.uuid for trace in trace_res.result]

            experiment_res = self.experiment_service.create_experiment(trace_uuids, experiment_name)
            if experiment_res.error or not experiment_res.result:
                return BaseResponse(error=experiment_res.error)

            return trace_res, experiment_res

        return trace_res
    
    def list_traces(self, uuid: Optional[str] = None) -> BaseResponse[List[TraceResponse]]:
        """List traces from the TSP server.

        Args:
            uuid (str, optional): The UUID of the trace to get. If None, all traces will be returned. Defaults to None.

        Returns:
            BaseResponse[List[TraceResponse]]: A response containing the traces' information.
        """

        return self.trace_service.list_traces(uuid)
