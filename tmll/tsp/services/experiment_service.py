from typing import Any, Dict, List, Optional, Union

from tmll.tsp.services.tsp_service import TSPService
from tmll.tsp.utils.pattern_extractor import PatternExtractor


class ExperimentService(TSPService):

    def create_experiment(self, trace_uuids: List[str], name: str) -> Union[Dict[str, str], Dict[str, Union[str, int]]]:
        """
        Create an experiment with the given traces.
        Return the experiment UUID, name, start time, end time, and number of events (i.e., its features).
        """

        # Create an experiment with the given traces
        command = ["python3", self.tsp_client_name,
                   "--open-experiment", name, "--uuids", *trace_uuids]
        execution = self.run_tsp_command(command)
        if "error" in execution:
            return execution

        process_output = execution["output"]

        # Just get the second line of the output (all the information about the experiment is in the second line)
        process_output = process_output.splitlines()[2]

        # Extract the experiment UUID, name, start time, end time, and number of events
        experiment = PatternExtractor.extract_trace_experiment_features(
            process_output)
        if experiment is None:
            return {"error": f"Error extracting features for experiment {name}"}

        return experiment

    def list_experiments(self, uuid: Optional[str] = None) -> Union[Dict[str, str], List]:
        """
        Get the list of experiments.
        If a UUID is given, get a single-item list with the experiment details.
        """

        # Get the list of experiments
        # If there was an error, return an empty dictionary along with the error message
        command = ["python3", self.tsp_client_name]
        if uuid is not None:
            command.extend(["--list-experiment", uuid])
        else:
            command.append("--list-experiments")

        executions = self.run_tsp_command(command)
        if "error" in executions:
            return executions

        process_output = executions["output"]

        experiments = []

        experiment_texts = PatternExtractor.extract_experiments(process_output)
        for experiment_text in experiment_texts:
            lines = experiment_text.split('\n')

            experiment_info = PatternExtractor.extract_trace_experiment_features(
                lines[0])
            if experiment_info:
                exp: dict[str, Any] = experiment_info.copy()

                # Skipping the first line which is experiment and the "Trace(s):" line
                for line in lines[2:]:
                    trace_info = PatternExtractor.extract_trace_experiment_features(
                        line)
                    if trace_info:
                        exp["traces"] = exp.get("traces", [])
                        exp["traces"].append(trace_info)

                experiments.append(exp)

        return experiments
