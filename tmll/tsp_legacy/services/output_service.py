from typing import List

from tmll.tsp_legacy.models.response.base import BaseResponse
from tmll.tsp_legacy.models.response.output import OutputResponse
from tmll.tsp_legacy.services.tsp_service import TSPService
from tmll.tsp_legacy.utils.pattern_extractor import PatternExtractor


class OutputService(TSPService):

    def list_outputs(self, uuid: str) -> BaseResponse[List[OutputResponse]]:
        """Get the list of outputs for the given UUID.

        Args:
            uuid (str): The UUID to get the outputs from.

        Returns:
            BaseResponse[List[OutputResponse]]: The list of outputs for the given UUID.
        """

        # Get the list of outputs for the given experiment UUID
        command = ["python3", self.tsp_client_name, "--list-outputs", uuid]
        execution = self.run_tsp_command(command)
        if execution.error or execution.result is None:
            return BaseResponse(error=execution.error)

        process_output = execution.result

        outputs = []
        for output in process_output.splitlines():
            # Extract the output type name and ID
            extracted_features = PatternExtractor.extract_output_features(output)
            if extracted_features is None:
                return BaseResponse(error=f"Error extracting features for output {output}")

            outputs.append(extracted_features)

        return BaseResponse(result=outputs)
