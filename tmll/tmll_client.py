"""
Trace Server Protocol (TSP) Machine Learning Library (TMLL) is a Python-based library that allows users to apply various machine learning techniques on the analyses of TSP.
The library is implemented as a set of Python classes that can be used to interact with Trace Server Protocol (TSP) and apply machine learning techniques on the data.
"""
import time
import re

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
from tmll.tsp.tsp.response import ResponseStatus
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

    def fetch_outputs(self, custom_output_ids: Optional[List[str]], force_reload: bool = False) -> None:
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

                # Check if the custom_output_ids is specified and the output is not in the custom_output_ids
                if custom_output_ids and output.id not in custom_output_ids:
                    continue

                # Get the trees of the outputs
                match output.type:
                    case "TABLE" | "DATA_TREE":
                        response = self.tsp_client.fetch_datatree(exp_uuid=self.experiment.uuid, output_id=output.id)
                        if response.status_code != 200:
                            response = self.tsp_client.fetch_timegraph_tree(exp_uuid=self.experiment.uuid, output_id=output.id)
                            if response.status_code != 200:
                                self.logger.error(f"Failed to fetch data tree. Error: {response.status_text}")
                                continue

                    case "TIME_GRAPH":
                        response = self.tsp_client.fetch_timegraph_tree(exp_uuid=self.experiment.uuid, output_id=output.id)
                        if response.status_code != 200:
                            self.logger.error(f"Failed to fetch time graph tree. Error: {response.status_text}")
                            continue

                    case "TREE_TIME_XY":
                        while True:
                            response = self.tsp_client.fetch_xy_tree(exp_uuid=self.experiment.uuid, output_id=output.id)
                            if response.status_code != 200:
                                self.logger.error(f"Failed to fetch XY tree. Error: {response.status_text}")
                                break
                            
                            # Wait until the model is completely fetched (i.e., status is COMPLETED)
                            if response.model.status.name == ResponseStatus.COMPLETED.name:
                                break

                            time.sleep(1)
                    case _:
                        self.logger.warning(f"Output type '{output.type}' is not supported.")
                        continue

                model = response.model.model
                if model is None:
                    self.logger.warning(f"Tree of the output '{output.name}' is None.")
                    continue

                tree = Tree.from_tsp_tree(model)

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
                    item_ids = list(map(int, [node.id for node in output["tree"].nodes]))
                    time_range_start = kwargs.get("start", self.experiment.start)
                    time_range_end = kwargs.get("end", self.experiment.end)
                    time_range_num_times = kwargs.get("num_times", 65536)

                    while True:
                        parameters = {
                            TspClient.PARAMETERS_KEY: {
                                TspClient.REQUESTED_ITEM_KEY: item_ids,
                                TspClient.REQUESTED_TIME_RANGE_KEY: {
                                    TspClient.REQUESTED_TIME_RANGE_START_KEY: time_range_start,
                                    TspClient.REQUESTED_TIME_RANGE_END_KEY: time_range_end,
                                    TspClient.REQUESTED_TIME_RANGE_NUM_TIMES_KEY: time_range_num_times
                                }
                            }
                        }

                        # Send a request to the TSP server to fetch the XY data
                        data = self.tsp_client.fetch_xy(exp_uuid=self.experiment.uuid, output_id=output["output"].id, parameters=parameters)
                        if data.status_code != 200 or data.model.model is None:
                            self.logger.error(f"Failed to fetch XY data. Error: {data.status_text}")
                            break  # Exit the loop if there's an error

                        if not data.model.model.series:
                            break  # Exit the loop if no more data is returned

                        x, y = None, None
                        for serie in data.model.model.series[:1]:
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
                            datasets[output["output"].id][series_name] = pd.concat([datasets[output["output"].id].get(series_name, pd.DataFrame()), dataset])

                        # Update the time_range_start for the next iteration
                        if x and len(x) > 0:
                            time_range_start = x[-1] + 1  # Start from the next timestamp after the last received

                            # Check if the time_range_start is greater than the time_range_end
                            if time_range_start > time_range_end:
                                break
                        else:
                            break  # Exit if no data was received in this iteration

                case "TIME_GRAPH":
                    self.logger.warning("Time graph output is not supported yet in the library.")
                    continue

                case "TABLE" | "DATA_TREE":
                    columns = self.tsp_client.fetch_virtual_table_columns(exp_uuid=self.experiment.uuid, output_id=output["output"].id)
                    if columns.status_code != 200:
                        self.logger.error(f"Failed to fetch '{output['output'].name}' virtual table columns. Error: {columns.status_text}")
                        continue

                    columns = [TableDataColumn.from_tsp_table_column(column) for column in columns.model.model.columns]

                    # We keep the initial columns to use them in the DataFrame creation
                    initial_table_columns = []

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
                            initial_table_columns = [c.name for c_id in table.columns for c in columns if c.id == c_id]
                            if not initial_table_columns:
                                initial_table_columns = [f"Column {c}" for c in table.columns]

                            datasets[output["output"].id] = pd.DataFrame(columns=initial_table_columns)

                        # If there are no rows in the table, break the loop since there is no more data to fetch
                        if not table.rows:
                            break

                        # Convert the table rows to a DataFrame
                        row_data = pd.DataFrame.from_dict({row.index: row.values for row in table.rows}, orient='index', columns=initial_table_columns)

                        # If the 'separate_columns' parameter is True, extract the features from the columns
                        # For example, if the column contains "key=value" pairs, extract the key and value as separate columns
                        separate_columns = kwargs.get("separate_columns", False)
                        if separate_columns:
                            row_data = self.extract_features_from_columns(row_data)

                        # Concatenate the row data to the DataFrame of the output
                        datasets[output["output"].id] = pd.concat([datasets[output["output"].id], row_data])

                        # If the number of rows in the table is less than the line count, break the loop since there is no more data to fetch
                        if len(table.rows) < line_count:
                            break
                        
                        # Update the start index for the next iteration (i.e., next batch of data)
                        start_index += line_count

                        break
                case _:
                    self.logger.warning(f"Output type '{output['output'].type}' is not supported.")
                    continue

            self.logger.info(f"Data fetched for the output '{output['output'].name}'.")

        self.logger.info("All data fetched successfully.")

        return datasets
    
    def extract_features_from_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy()

        for column in df.columns:
            new_columns = {}

            for row_index, row in enumerate(df[column].astype(str).replace("\n", "").str.strip().str.split(", ")):
                for part in row:
                    if "=" in part:
                        key, value = part.split("=")
                        key = re.sub(r"[^\w\s]", "", key).strip()
                        new_col_name = f"{column}_{key}"

                        if new_col_name not in new_columns:
                            new_columns[new_col_name] = [np.nan] * len(df)
                        
                        new_columns[new_col_name][row_index] = value
                    
            for col_name, values in new_columns.items():
                df[col_name] = pd.Series(values, index=df.index)

            if new_columns:
                df.drop(columns=[column], inplace=True)

        return df
