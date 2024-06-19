from typing import Dict, List, Union
from tmll.tsp.services.tsp_service import TSPService
from tmll.tsp.utils.pattern_extractor import PatternExtractor


class OutputService(TSPService):

    def list_outputs(self, uuid: str) -> Union[Dict[str, str], List[dict[str, str]]]:
        """
        Get the list of outputs (i.e., analysis types from TSP) for the given experiment UUID.
        """

        # Get the list of outputs for the given experiment UUID
        command = ["python3", self.tsp_client_name, "--list-outputs", uuid]
        execution = self.run_tsp_command(command)
        if "error" in execution:
            return execution

        process_output = execution["output"]

        outputs = []
        for output in process_output.splitlines():
            # Extract the output type name and ID
            extracted_features = PatternExtractor.extract_output_features(
                output)
            if "error" in extracted_features:
                return extracted_features

            outputs.append(extracted_features)

        return outputs
