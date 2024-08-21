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

from tmll.utils.name_generator import NameGenerator

from tmll.common.services.logger import Logger


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
                self.logger.error(f"TSP server is not running properly. Check its health status.")
                return
        except Exception as e:
            # If the TSP server is not running and the user has not specified to install the TSP server, return
            if not install_tsp_server:
                self.logger.error(
                    "Failed to connect to the TSP server. Please make sure that the TSP server is running. If you want to install the TSP server, set the 'install_tsp_server' parameter to True.")
                return
            
            self.logger.warning("TSP server is not running. Trying to run the TSP server, and if not installed, install it.")

            tsp_installer = TSPInstaller()
            tsp_installer.install()

            # Check if the TSP server is installed successfully
            response = self.tsp_client.fetch_health()
            if response.status_code != 200:
                self.logger.error("Failed to install the TSP server.")
                return

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

    def import_traces(self, traces: List[Dict[str, str]], experiment_name: str, remove_previous: bool = False) -> None:
        """
        Import traces into the Trace Server Protocol (TSP) server.

        Steps:
            1. Open the traces
            2. Create an experiment with the opened traces
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
                self.logger.error("The 'path' parameter is required for each trace.")
                continue

            # If the name is not provided for the trace, generate a name for the trace
            if "name" not in trace:
                trace["name"] = NameGenerator.generate_name(base=trace["path"])
                self.logger.info(f"Generating a name for the trace: {trace['name']}")

            response = self.tsp_client.open_trace(name=trace["name"], path=trace["path"])
            if response.status_code != 200:
                self.logger.error(f"Failed to open trace '{trace['name']}'. Error: {response.status_text}")
                continue

            self.traces.append(Trace.from_tsp_trace(response.model))
            self.logger.info(f"Trace '{trace['name']}' opened successfully.")

        # If create_experiment is True, create an experiment with the opened traces
        trace_uuids = [trace.uuid for trace in self.traces]

        opened_experiment = self.tsp_client.open_experiment(name=experiment_name, traces=trace_uuids)
        if opened_experiment.status_code != 200:
            self.logger.error(f"Failed to open experiment '{experiment_name}'. Error: {opened_experiment.status_text}")
            return

        # Check the status of the experiment periodically until it is completed
        # P.S.: It's not an efficient way to check the status. However, the tsp server does not provide a way to get the status of the experiment directly.
        self.logger.info(f"Checking the indexing status of the experiment '{experiment_name}'.")
        while (True):
            status = self.tsp_client.fetch_experiment(opened_experiment.model.UUID)
            if status.status_code != 200:
                self.logger.error(f"Failed to fetch experiment. Error: {status.status_text}")
                return

            if status.model.indexing_status.name == IndexingStatus.COMPLETED.name:
                self.experiment = Experiment.from_tsp_experiment(status.model)
                self.logger.info(f"Experiment '{experiment_name}' is loaded completely.")
                break

            # Wait for 1 second before checking the status again
            time.sleep(1)

    def fetch_outputs(self, force_reload: bool = False) -> None:
        if self.experiment is None:
            self.logger.error("Experiment is not loaded. Please load the experiment first by calling the 'import_traces' method.")
            return

        if not self.outputs or force_reload:
            # Get the outputs of the experiment
            outputs = self.tsp_client.fetch_experiment_outputs(self.experiment.uuid)
            if outputs.status_code != 200:
                self.logger.error(f"Failed to fetch experiment outputs. Error: {outputs.status_text}")
                return

            self.outputs = []  # Reset the outputs
            for output_ in outputs.model.descriptors:
                output = Output.from_tsp_output(output_)

                # Get the trees of the outputs
                match output.type:
                    case "TABLE" | "DATA_TREE":
                        tree = self.tsp_client.fetch_datatree(exp_uuid=self.experiment.uuid, output_id=output.id)
                        if tree.status_code != 200:
                            tree = self.tsp_client.fetch_timegraph_tree(exp_uuid=self.experiment.uuid, output_id=output.id)
                            if tree.status_code != 200:
                                self.logger.error(f"Failed to fetch data tree. Error: {tree.status_text}")
                                continue

                    case "TIME_GRAPH":
                        tree = self.tsp_client.fetch_timegraph_tree(exp_uuid=self.experiment.uuid, output_id=output.id)
                        if tree.status_code != 200:
                            self.logger.error(f"Failed to fetch time graph tree. Error: {tree.status_text}")
                            continue

                    case "TREE_TIME_XY":
                        tree = self.tsp_client.fetch_xy_tree(exp_uuid=self.experiment.uuid, output_id=output.id)
                        if tree.status_code != 200:
                            self.logger.error(f"Failed to fetch XY tree. Error: {tree.status_text}")
                            continue

                    case _:
                        self.logger.warning(f"Output type '{output.type}' is not supported.")
                        continue

                tree = tree.model.model
                if tree is None:
                    self.logger.warning(f"Tree of the output '{output.name}' is None.")
                    continue

                tree = Tree.from_tsp_tree(tree)

                self.outputs.append({
                    "output": output,
                    "tree": tree
                })
                self.logger.info(f"Output '{output.name}' and its tree fetched successfully.")

            self.logger.info("Outputs are fetched successfully.")
        else:
            self.logger.info("Outputs are already fetched. If you want to force reload the outputs, set the 'force_reload' parameter to True.")

    def fetch_data(self, custom_output_ids: Optional[List[str]] = None, **kwargs) -> Union[None, Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]]:
        # Check if the experiment is loaded
        if self.experiment is None:
            self.logger.error("Experiment is not loaded. Please load the experiment first.")
            return None

        # Check if the outputs are fetched
        if not self.outputs:
            self.logger.error("Outputs are not fetched. Please fetch the outputs first.")
            return None

        datasets = {}
        for output in self.outputs:
            # If custom_output_ids is specified, only fetch the data of the specified outputs
            if custom_output_ids and output["output"].id not in custom_output_ids:
                continue

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
                        self.logger.error(f"Failed to fetch XY data. Error: {data.status_text}")
                        continue

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
                        datasets[output["output"].id] = datasets.get(output["output"].id, {})
                        datasets[output["output"].id][serie.series_name] = dataset

                case "TIME_GRAPH":
                    self.logger.warning("Time graph output is not supported yet in the library.")
                    continue

                case "TABLE" | "DATA_TREE":
                    columns = self.tsp_client.fetch_virtual_table_columns(exp_uuid=self.experiment.uuid, output_id=output["output"].id)
                    if columns.status_code != 200:
                        self.logger.error(f"Failed to fetch '{output['output'].name}' virtual table columns. Error: {columns.status_text}")
                        continue

                    columns = [TableDataColumn.from_tsp_table_column(column) for column in columns.model.model.columns]

                    start_index = int(kwargs.get("table_line_start_index", 0)) # Start index of the table data. Default is 0 (i.e., the first row)
                    line_count = int(kwargs.get("table_line_count", 65536)) # 65536 is the maximum value that the TSP server accepts
                    column_ids = list(map(int, kwargs.get("table_line_column_ids", []))) # Which columns to fetch from the table
                    search_direction = kwargs.get("table_line_search_direction", "NEXT") # Search direction for the table data (i.e., NEXT or PREVIOUS)
                    while True:
                        # Prepare the parameters for the TSP server
                        parameters = {
                            TspClient.PARAMETERS_KEY: {
                                TspClient.REQUESTED_TABLE_LINE_INDEX_KEY: start_index,
                                TspClient.REQUESTED_TABLE_LINE_COUNT_KEY: line_count,
                                TspClient.REQUESTED_TABLE_LINE_COLUMN_IDS_KEY: column_ids,
                                TspClient.REQUESTED_TABLE_LINE_SEACH_DIRECTION_KEY: search_direction
                            }
                        }

                        # Send a request to the TSP server to fetch the virtual table data
                        table_request = self.tsp_client.fetch_virtual_table_lines(exp_uuid=self.experiment.uuid, output_id=output["output"].id, parameters=parameters)

                        # If the request is not successful or the table data is None, break the loop
                        if table_request.status_code != 200 or table_request.model.model is None:
                            self.logger.error(f"Failed to fetch '{output['output'].name}' virtual table data. Error: {table_request.status_text}")
                            break

                        # Create the table model
                        table = TableData.from_tsp_table(table_request.model.model)

                        # If the DataFrame for the output is not created yet, create it
                        if output["output"].id not in datasets:
                            column_names = [c.name for c_id in table.columns for c in columns if c.id == c_id]
                            datasets[output["output"].id] = pd.DataFrame(columns=column_names)

                        # If there are no rows in the table, break the loop since there is no more data to fetch
                        if not table.rows:
                            break

                        # Iterate through the table and add each row to the DataFrame
                        row_data = pd.DataFrame.from_dict({row.index: row.values for row in table.rows}, orient='index')
                        datasets[output["output"].id] = pd.concat([datasets[output["output"].id], row_data])

                        # If the number of rows in the table is less than the line count, break the loop since there is no more data to fetch
                        if len(table.rows) < line_count:
                            break
                        
                        # Update the start index for the next iteration (i.e., next batch of data)
                        start_index += line_count
                case _:
                    self.logger.warning(f"Output type '{output['output'].type}' is not supported.")
                    continue

            self.logger.info(f"Data fetched for the output '{output['output'].name}'.")

        self.logger.info("All data fetched successfully.")

        return datasets
