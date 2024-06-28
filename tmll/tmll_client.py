"""
Trace Compass Machine Learning Library (TMLL) is a Python-based library that allows users to apply various machine learning techniques on the analyses of Trace Compass.
The library is implemented as a set of Python classes that can be used to interact with Trace Compass Server Protocol (TSP) and apply machine learning techniques on the data.
"""
import time

from typing import Dict, List, Optional

from tmll.common.models.output import Output
from tmll.common.models.trace import Trace
from tmll.common.models.experiment import Experiment
from tmll.common.models.tree.tree import Tree
from tmll.tsp.tsp import experiment
from tmll.tsp.tsp.tsp_client import TspClient

from tmll.ml.unsupervised.clustering import Clustering

from tmll.utils.name_generator import NameGenerator

from tmll.common.services.logger import Logger
from tmll.common.constants import TSP as TSP_CONSTANTS


class TMLLClient:

    def __init__(self, tsp_server_host: str, tsp_server_port: int, verbose: bool = True) -> None:
        self.tsp_client = TspClient(f"http://{tsp_server_host}:{tsp_server_port}/tsp/api/")

        self.traces = []
        self.experiment = None
        self.outputs = []

        self.logger = Logger("TMLLClient", verbose)

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
        Import traces into the Trace Compass Server Protocol (TSP) server.

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
        while (True):
            self.logger.info(f"Checking the status of the experiment '{experiment_name}'.")

            status = self.tsp_client.fetch_experiment(opened_experiment.model.UUID)
            if status.status_code != 200:
                raise Exception(f"Failed to fetch experiment. Error: {status.status_text}")

            if status.model.indexin_status == "COMPLETED":
                self.experiment = Experiment.from_tsp_experiment(status.model)
                self.logger.info(f"Experiment '{experiment_name}' is loaded completely.")
                break

            # Wait for 5 seconds before checking the status again
            time.sleep(5)

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
