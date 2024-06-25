"""
Trace Compass Machine Learning Library (TMLL) is a Python-based library that allows users to apply various machine learning techniques on the analyses of Trace Compass.
The library is implemented as a set of Python classes that can be used to interact with Trace Compass Server Protocol (TSP) and apply machine learning techniques on the data.
"""

from tmll.tsp_legacy.services.tree_service import TreeService
from tmll.tsp_legacy.services.data_service import DataService
from tmll.tsp_legacy.services.output_service import OutputService
from tmll.tsp_legacy.services.experiment_service import ExperimentService
from tmll.tsp_legacy.services.trace_service import TraceService

TSP_CLIENT_NAME = "tsp_cli_client"


class TMLLClient:

    def __init__(self, tsp_client_path: str) -> None:
        self.tsp_client_path = tsp_client_path
        self.tsp_client_name = TSP_CLIENT_NAME
