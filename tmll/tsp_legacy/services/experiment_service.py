from typing import List, Optional

from tmll.tsp_legacy.models.response.base import BaseResponse
from tmll.tsp_legacy.models.response.experiment import ExperimentResponse
from tmll.tsp_legacy.services.tsp_service import TSPService
from tmll.tsp_legacy.utils.pattern_extractor import PatternExtractor


class ExperimentService(TSPService):

    def create_experiment(self, trace_uuids: List[str], name: str) -> BaseResponse[ExperimentResponse]:
        """Create an experiment with the given traces.

        Args:
            trace_uuids (List[str]): A list of trace UUIDs.
            name (str): The name of the experiment.

        Returns:
            BaseResponse[ExperimentResponse]: The experiment response containing the UUID, name, start time, end time, and number of events.
        """

        # Create an experiment with the given traces
        command = ["python3", self.tsp_client_name, "--open-experiment", name, "--uuids", *trace_uuids]
        execution = self.run_tsp_command(command)
        if execution.error or execution.result is None:
            return BaseResponse(error=execution.error)

        process_output = execution.result

        # Just get the second line of the output (all the information about the experiment is in the second line)
        process_output = process_output.splitlines()[2]

        # Extract the experiment UUID, name, start time, end time, and number of events
        experiment = PatternExtractor.extract_trace_experiment_features(process_output)
        if experiment is None:
            return BaseResponse(error=f"Error extracting features for experiment {name}")

        return BaseResponse(result=ExperimentResponse.from_trace_response(experiment))

    def list_experiments(self, uuid: Optional[str] = None) -> BaseResponse[List[ExperimentResponse]]:
        """Get the list of experiments from the TSP server.

        Args:
            uuid (Optional[str], optional): The UUID of the experiment to get. If None, all experiments will be returned. Defaults to None.

        Returns:
            BaseResponse[List[ExperimentResponse]]: The list of experiments from the TSP server.
        """

        # Get the list of experiments
        # If there was an error, return an empty dictionary along with the error message
        command = ["python3", self.tsp_client_name]
        if uuid is not None:
            command.extend(["--list-experiment", uuid])
        else:
            command.append("--list-experiments")

        execution = self.run_tsp_command(command)
        if execution.error or execution.result is None:
            return BaseResponse(error=execution.error)

        process_output = execution.result

        experiments = []

        experiment_texts = PatternExtractor.extract_experiments(process_output)
        for experiment_text in experiment_texts:
            lines = experiment_text.split('\n')

            experiment_info = PatternExtractor.extract_trace_experiment_features(lines[0])
            if experiment_info:
                exp = ExperimentResponse.from_trace_response(experiment_info)

                # Skipping the first line which is experiment and the "Trace(s):" line
                for line in lines[2:]:
                    trace_info = PatternExtractor.extract_trace_experiment_features(line)
                    if trace_info:
                        exp.traces.append(trace_info)

                experiments.append(exp)

        return BaseResponse(result=experiments)
