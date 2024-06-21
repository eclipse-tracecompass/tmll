from typing import Any, Dict, Optional, Union

from tmll.tsp.services.tsp_service import TSPService
from tmll.tsp.utils.pattern_extractor import PatternExtractor


class TraceService(TSPService):

    def add_traces(self, traces: Dict[str, str]) -> Dict[str, Union[str, Any]]:
        """Add traces to the TSP server.

        Args:
            traces (Dict[str, str]): A dictionary of trace names and their paths.

        Returns:
            Dict[str, Union[str, Any]]: A dictionary containing the traces and their features (e.g., UUID, name, start time, end time, number of events).
        """

        output = {}

        # Add traces to the TSP server
        generated_traces = {}
        for trace_name, trace_path in traces.items():
            # Run the TSP client to add the trace to the TSP server
            command = ["python3", self.tsp_client_name, "--open-trace", trace_path, "--name", trace_name]
            execution = self.run_tsp_command(command)

            if "error" in execution:
                return execution

            process_output = execution["output"]

            # Extract the trace features
            trace_features = PatternExtractor.extract_trace_experiment_features(process_output)
            if trace_features is None:
                return {"error": f"Error extracting features for trace {trace_name}"}

            generated_traces[trace_name] = trace_features

        # Get the list of created traces and their UUIDs
        if not generated_traces:
            return {"error": "No traces were created."}

        output["traces"] = generated_traces

        return output

    def list_traces(self, uuid: Optional[str] = None) -> Dict[str, Union[str, Any]]:
        """Get the list of traces from the TSP server.

        Args:
            uuid (Optional[str], optional): The UUID of the trace to get. If None, all traces will be returned. Defaults to None.

        Returns:
            Dict[str, Union[str, Any]]: A dictionary containing the traces and their features (e.g., UUID, name, start time, end time, number of events).
        """

        # Get the list of traces
        # If a specific UUID is given, get the corresponding trace for that UUID
        command = ["python3", self.tsp_client_name]
        if uuid is not None:
            command.extend(["--list-trace", uuid])
        else:
            command.append("--list-traces")

        execution = self.run_tsp_command(command)
        if "error" in execution:
            return execution

        process_output = execution["output"]

        traces = process_output.splitlines()

        output = {}
        for trace in traces:
            # Extract the trace features
            extracted_features = PatternExtractor.extract_trace_experiment_features(trace)
            if extracted_features is None:
                return {"error": f"Error extracting features for trace {trace}"}

            output["traces"] = output.get("traces", [])
            output["traces"].append(extracted_features)

        return output
