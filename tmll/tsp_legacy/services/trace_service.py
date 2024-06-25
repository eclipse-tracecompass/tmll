from typing import Dict, List, Optional

from tmll.tsp_legacy.models.response.base import BaseResponse
from tmll.tsp_legacy.models.response.trace import TraceResponse
from tmll.tsp_legacy.services.tsp_service import TSPService
from tmll.tsp_legacy.utils.pattern_extractor import PatternExtractor


class TraceService(TSPService):

    def add_traces(self, traces: Dict[str, str]) -> BaseResponse[List[TraceResponse]]:
        """Add traces to the TSP server.

        Args:
            traces (Dict[str, str]): A dictionary of trace names and their paths.

        Returns:
            BaseResponse[List[TraceResponse]]: The list of traces and their features (e.g., UUID, name, start time, end time, number of events).
        """

        # Add traces to the TSP server
        generated_traces = []
        for trace_name, trace_path in traces.items():
            # Run the TSP client to add the trace to the TSP server
            command = ["python3", self.tsp_client_name, "--open-trace", trace_path, "--name", trace_name]
            execution = self.run_tsp_command(command)

            if execution.error or execution.result is None:
                return BaseResponse(error=execution.error)

            process_output = execution.result

            # Extract the trace features
            trace = PatternExtractor.extract_trace_experiment_features(process_output)
            if trace is None:
                return BaseResponse(error=f"Error extracting features for trace {trace_name}")

            generated_traces.append(trace)

        # Get the list of created traces and their UUIDs
        if not generated_traces:
            return BaseResponse(error="No traces were created.")

        return BaseResponse(result=generated_traces)

    def list_traces(self, uuid: Optional[str] = None) -> BaseResponse[List[TraceResponse]]:
        """Get the list of traces from the TSP server.

        Args:
            uuid (Optional[str], optional): The UUID of the trace to get. If None, all traces will be returned. Defaults to None.

        Returns:
            BaseResponse[List[TraceResponse]]: The list of traces with their features from the TSP server.
        """

        # Get the list of traces
        # If a specific UUID is given, get the corresponding trace for that UUID
        command = ["python3", self.tsp_client_name]
        if uuid is not None:
            command.extend(["--list-trace", uuid])
        else:
            command.append("--list-traces")

        execution = self.run_tsp_command(command)

        if execution.error or execution.result is None:
            return BaseResponse(error=execution.error)

        process_output = execution.result

        traces = []
        for trace in process_output.splitlines():
            # Extract the trace features
            extracted_features = PatternExtractor.extract_trace_experiment_features(trace)
            if extracted_features is None:
                return BaseResponse(error=f"Error extracting features for trace {trace}")

            traces.append(extracted_features)

        return BaseResponse(result=traces)
