"""
Trace Server Protocol (TSP) Machine Learning Library (TMLL) is a Python-based library that allows users to apply various machine learning techniques on the analyses of TSP.
The library is implemented as a set of Python classes that can be used to interact with Trace Server Protocol (TSP) and apply machine learning techniques on the data.
"""
import time

import pandas as pd
import numpy as np

from typing import Dict, List, Optional, Union

from tmll.common.models.data.table.column import TableDataColumn
from tmll.common.models.data.table.table import TableData
from tmll.common.models.output import Output
from tmll.common.models.trace import Trace
from tmll.common.models.experiment import Experiment
from tmll.common.models.tree.tree import Tree

from tmll.tsp.tsp.indexing_status import IndexingStatus
from tmll.tsp.tsp.tsp_client import TspClient

from tmll.services.tsp_installer import TSPInstaller

from tmll.ml.unsupervised.clustering import Clustering

from tmll.utils.name_generator import NameGenerator

from tmll.common.services.logger import Logger
from tmll.common.constants import TSP as TSP_CONSTANTS


class TMLLClient:

    def __init__(self, tsp_server_host: str = "localhost", tsp_server_port: int = 8080,
                 install_tsp_server: bool = True, force_install: bool = False,
                 verbose: bool = True) -> None:

        base_url = f"http://{tsp_server_host}:{tsp_server_port}/tsp/api/"
        self.tsp_client = TspClient(base_url=base_url)

        self.logger = Logger("TMLLClient", verbose)

        # Check if the TSP server is running (i.e., check if server is reachable)
        try:
            response = self.tsp_client.fetch_health()
            if response.status_code != 200:
                raise Exception("TSP server is not running properly. Check its health status.")
        except Exception as e:
            # If the TSP server is not running and the user has not specified to install the TSP server, raise an exception
            if not install_tsp_server:
                raise Exception(
                    "Failed to connect to the TSP server. Please make sure that the TSP server is running. If you want to install the TSP server, set the 'install_tsp_server' parameter to True.")

            self.logger.warning("TSP server is not running. Installing the TSP server.")

            tsp_installer = TSPInstaller()
            tsp_installer.install()

            # Check if the TSP server is installed successfully
            response = self.tsp_client.fetch_health()
            if response.status_code != 200:
                raise Exception("Failed to install the TSP server. Please check the logs for more information.")

        self.logger.info("Connected to the TSP server successfully.")

        self.traces = []
        self.experiment = None
        self.outputs = []

        """
        THESE ARE THE TEMPORARY METHODS THAT WILL BE REMOVED IN THE FINAL VERSION
        """
        def _delete_experiments():
            response = self.tsp_client.fetch_experiments()
            for experiment in response.model.experiments:
                self.tsp_client.delete_experiment(experiment.UUID)

        def _delete_traces():
            response = self.tsp_client.fetch_traces()
            for trace in response.model.traces:
                # Do not also delete the trace from disk; file part of this repo.
                self.tsp_client.delete_trace(trace.UUID, False)

        _delete_experiments()
        _delete_traces()

    def import_traces(self, traces: List[Dict[str, str]], experiment_name: Optional[str] = None, remove_previous: bool = True, **kwargs) -> None:
        """
        Import traces into the Trace Server Protocol (TSP) server.

        Steps:
            1. Open the traces
            2. Create an experiment with the opened traces (optional)
            3. Get the outputs of the experiment (optional)
            4. Get the trees of the outputs
            5. Fetch the results
        """

        # Remove the previous traces and experiment if remove_previous is True
        if remove_previous:
            self.logger.info("Removing previous traces and experiment.")
            self.traces = []
            self.experiment = None
            self.outputs = []

        # For each trace, add it to the TSP server
        for trace in traces:
            if "path" not in trace:
                raise ValueError("Each trace must have a 'path' key.")

            if "name" not in trace:
                if kwargs.get("generate_name", False):
                    trace["name"] = NameGenerator.generate_name(base=trace["path"])
                    self.logger.info(f"Generating a name for the trace: {trace['name']}")
                else:
                    raise ValueError("The 'name' parameter is required if 'generate_name' is False.")

            response = self.tsp_client.open_trace(name=trace["name"], path=trace["path"])
            if response.status_code != 200:
                raise Exception(f"Failed to open trace. Error: {response.status_text}")

            self.traces.append(Trace.from_tsp_trace(response.model))
            self.logger.info(f"Trace '{trace['name']}' opened successfully.")

        # If create_experiment is True, create an experiment with the opened traces
        trace_uuids = [trace.uuid for trace in self.traces]

        if experiment_name is None:
            if kwargs.get("generate_name", False):
                experiment_name = NameGenerator.generate_name(base="experiment", length=8)
                self.logger.info(f"Generating a name for the experiment: {experiment_name}")
            else:
                raise ValueError("The 'experiment_name' parameter is required if 'generate_name' is False.")

        opened_experiment = self.tsp_client.open_experiment(name=experiment_name, traces=trace_uuids)
        if opened_experiment.status_code != 200:
            raise Exception(f"Failed to open experiment. Error: {opened_experiment.status_text}")

        # Check the status of the experiment periodically until it is completed
        # P.S.: It's not an efficient way to check the status. However, the tsp server does not provide a way to get the status of the experiment directly.
        self.logger.info(f"Checking the indexing status of the experiment '{experiment_name}'.")
        while (True):
            status = self.tsp_client.fetch_experiment(opened_experiment.model.UUID)
            if status.status_code != 200:
                raise Exception(f"Failed to fetch experiment. Error: {status.status_text}")

            if status.model.indexing_status.name == IndexingStatus.COMPLETED.name:
                self.experiment = Experiment.from_tsp_experiment(status.model)
                self.logger.info(f"Experiment '{experiment_name}' is loaded completely.")
                break

            # Wait for 1 second before checking the status again
            time.sleep(1)

        # Get the outputs of the experiment
        outputs = self.tsp_client.fetch_experiment_outputs(self.experiment.uuid)
        if outputs.status_code != 200:
            raise Exception(f"Failed to fetch experiment outputs. Error: {outputs.status_text}")

        for output_ in outputs.model.descriptors:
            output = Output.from_tsp_output(output_)

            """
            Fetching all of the outputs is not a good idea since it requires a substantial amount of time and resources.
            Therefore, we only fetch the outputs that are specified by the user or the candidate outputs.

            Steps:
                1. If the user has specified "minimal_outputs" parameter to true, only fetch the candidate outputs (check the constants.py file). Otherwise, fetch all of the outputs.
                2. If the user has specified "outputs" parameter, only fetch the specified outputs.
                3. If the user has not specified any parameter, only fetch the candidate outputs.
            """
            # Step 1: Check if "minimal_outputs" parameter is true and "outputs" is not specified
            if "minimal_outputs" in kwargs and kwargs["minimal_outputs"] and "outputs" not in kwargs:
                # Only fetch candidate outputs
                if output.id not in TSP_CONSTANTS.CANDIDATE_OUTPUTS:
                    continue

            # Step 2: Check if "outputs" parameter is specified
            if "outputs" in kwargs:
                # Only fetch specified outputs
                if output.id not in kwargs["outputs"]:
                    continue

            # Step 3: If no parameter is specified, only fetch candidate outputs
            if "minimal_outputs" not in kwargs and "outputs" not in kwargs:
                # Only fetch candidate outputs
                if output.id not in TSP_CONSTANTS.CANDIDATE_OUTPUTS:
                    continue

            # Get the trees of the outputs
            match output.type:
                case "TABLE" | "DATA_TREE":
                    tree = self.tsp_client.fetch_datatree(exp_uuid=self.experiment.uuid, output_id=output.id)
                    if tree.status_code != 200:
                        tree = self.tsp_client.fetch_timegraph_tree(exp_uuid=self.experiment.uuid, output_id=output.id)
                        if tree.status_code != 200:
                            raise Exception(f"Failed to fetch data tree or time graph tree. Error: {tree.status_text}")

                case "TIME_GRAPH":
                    tree = self.tsp_client.fetch_timegraph_tree(exp_uuid=self.experiment.uuid, output_id=output.id)
                    if tree.status_code != 200:
                        raise Exception(f"Failed to fetch time graph tree. Error: {tree.status_text}")

                case "TREE_TIME_XY":
                    tree = self.tsp_client.fetch_xy_tree(exp_uuid=self.experiment.uuid, output_id=output.id)
                    if tree.status_code != 200:
                        raise Exception(f"Failed to fetch XY tree. Error: {tree.status_text}")

                case _:
                    raise ValueError(f"Output type '{output.type}' is not supported.")

            tree = tree.model.model
            if tree is None:
                continue

            tree = Tree.from_tsp_tree(tree)

            self.outputs.append({
                "output": output,
                "tree": tree
            })
            self.logger.info(f"Output '{output.name}' and its tree fetched successfully.")

    def apply_clustering(self, with_results: bool = True, **kwargs) -> Union[None, Dict]:
        """
        Apply clustering on the outputs of the experiment.
        """

        # Check if the experiment is loaded
        if self.experiment is None:
            raise ValueError("Experiment is not loaded. Please load the experiment first.")

        for output in self.outputs:
            datasets = {}

            match output["output"].type:
                case "TREE_TIME_XY":
                    # Prepare the parameters for the TSP server
                    parameters = {
                        TspClient.PARAMETERS_KEY: {
                            TspClient.REQUESTED_ITEM_KEY: list(map(int, [node.id for node in output["tree"].nodes])),
                            TspClient.REQUESTED_TIME_RANGE_KEY: {
                                TspClient.REQUESTED_TIME_RANGE_START_KEY: kwargs.get("start", self.experiment.start),
                                TspClient.REQUESTED_TIME_RANGE_END_KEY: kwargs.get("end", self.experiment.end),
                                TspClient.REQUESTED_TIME_RANGE_NUM_TIMES_KEY: kwargs.get("num_times", 1000)
                            }
                        }
                    }

                    # Send a request to the TSP server to fetch the XY data
                    data = self.tsp_client.fetch_xy(exp_uuid=self.experiment.uuid, output_id=output["output"].id, parameters=parameters)
                    if data.status_code != 200 or data.model.model is None:
                        raise Exception(f"Failed to fetch XY data. Error: {data.status_text}")

                    for serie in data.model.model.series:
                        # Get the x and y values from the data
                        x, y = serie.x_values, serie.y_values

                        # Create a pandas DataFrame from the data
                        dataset = pd.DataFrame(data=np.c_[x, y], columns=["x", "y"])

                        # Get the series name
                        series_name = serie.series_name
                        if not series_name or series_name == "":
                            series_name = f"Series {serie.series_id}"

                        # Add the dataset to the datasets dictionary
                        datasets[serie.series_name] = dataset

                case "TIME_GRAPH":
                    self.logger.info("Time graph output is not supported yet in the library.")
                    continue

                case "TABLE":
                    columns = self.tsp_client.fetch_virtual_table_columns(exp_uuid=self.experiment.uuid, output_id=output["output"].id)
                    if columns.status_code != 200:
                        raise Exception(f"Failed to fetch virtual table columns. Error: {columns.status_text}")
                    columns = [TableDataColumn.from_tsp_table_column(column) for column in columns.model.model.columns]

                    parameters = {
                        TspClient.PARAMETERS_KEY: {
                            TspClient.REQUESTED_TABLE_LINE_INDEX_KEY: int(kwargs.get("table_line_index", 0)),
                            TspClient.REQUESTED_TABLE_LINE_COUNT_KEY: int(kwargs.get("table_line_count", 5000)),
                            TspClient.REQUESTED_TABLE_LINE_COLUMN_IDS_KEY: list(map(int, kwargs.get("table_line_column_ids", []))),
                            TspClient.REQUESTED_TABLE_LINE_SEACH_DIRECTION_KEY: kwargs.get("table_line_search_direction", "NEXT")
                        }
                    }

                    table = self.tsp_client.fetch_virtual_table_lines(exp_uuid=self.experiment.uuid, output_id=output["output"].id, parameters=parameters)
                    if table.status_code != 200 or table.model.model is None:
                        raise Exception(f"Failed to fetch virtual table lines. Error: {table.status_text}")

                    table = TableData.from_tsp_table(table.model.model)

                    # Get the column names of the dataset
                    column_names = [c.name for c_id in table.columns for c in columns if c.id == c_id]

                    # Create an empty DataFrame
                    dataset = pd.DataFrame(columns=column_names)

                    # Iterate through the table, and add each row to the DataFrame
                    for row in table.rows:
                        dataset.loc[row.index] = row.values

                    # Add the dataset to the datasets dictionary
                    datasets[output["output"].name] = dataset
                case _:
                    raise ValueError(f"Output type '{output['output'].type}' is not supported.")

            # Apply clustering on the datasets
            for dataset_name, dataset in datasets.items():
                # Apply clustering on the data
                clustering = Clustering(dataset, normalize=True)
                results = clustering.execute()

                # Add the results to the output dictionary (if results do not exist, create a new dictionary)
                if "results" not in output:
                    output["results"] = {}
                output["results"][dataset_name] = results

                self.logger.info(f"Clustering applied on the dataset '{dataset_name}'.")

        # If user has specified to return the results, return the results
        if with_results:
            return {
                "experiment": self.experiment,
                "outputs": self.outputs
            }

        return None
