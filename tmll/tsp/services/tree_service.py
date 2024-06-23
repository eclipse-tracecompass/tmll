from tmll.tsp.models.response.base import BaseResponse
from tmll.tsp.models.response.tree.table import TableTreeResponse
from tmll.tsp.models.response.tree.xy import XYTreeResponse
from tmll.tsp.services.tsp_service import TSPService
from tmll.tsp.utils.pattern_extractor import PatternExtractor

AVAILABLE_TREE_TYPES = ["xy", "timegraph", "table"]


class TreeService(TSPService):

    def get_tree(self, uuid: str, output_id: str, tree_type: str) -> BaseResponse:
        """Get the tree for the given experiment UUID and output type.

        Args:
            uuid (str): The UUID which the tree belongs to.
            output_id (str): The output ID which the tree belongs to.
            tree_type (str): The type of tree to get. You can use get_tree_types() to get the available tree types.

        Returns:
            BaseResponse: The tree for the given UUID and output type.
        """

        # Get the tree for the given experiment UUID and output type
        if tree_type == "xy":
            return self.__get_tree_xy(uuid, output_id)
        elif tree_type == "timegraph":
            return self.__get_tree_timegraph(uuid, output_id)
        elif tree_type == "data":
            return self.__get_tree(uuid, output_id)
        else:
            return BaseResponse(error=f"Invalid tree type. Available tree types: {AVAILABLE_TREE_TYPES}")

    @staticmethod
    def get_tree_types() -> list[str]:
        """Get the available tree types.

        Returns:
            list[str]: The available tree types.
        """
        return AVAILABLE_TREE_TYPES

    def __get_tree_xy(self, uuid: str, output_id: str) -> BaseResponse[XYTreeResponse]:
        """Get the XY tree for the given experiment UUID and output type.

        Args:
            uuid (str): The UUID which the tree belongs to.
            output_id (str): The output ID which the tree belongs to.

        Returns:
            BaseResponse[XYTreeResponse]: The XY tree for the given UUID and output type.
        """

        # Get the XY tree for the given experiment UUID and output type
        tree = self.__get_tree(uuid, output_id)
        if tree.error or tree.result is None:
            return BaseResponse(error=tree.error)

        return BaseResponse(result=XYTreeResponse.from_table_tree_response(tree.result))

    def __get_tree_timegraph(self, uuid: str, output_id: str) -> BaseResponse:
        """(NOT IMPLEMENTED YET!) Get the timegraph tree for the given experiment UUID and output type.

        Args:
            uuid (str): The UUID which the tree belongs to.
            output_id (str): The output ID which the tree belongs to.

        Returns:
            BaseResponse: The timegraph tree for the given UUID and output type.
        """

        # Not implemented yet
        return BaseResponse(error="Timegraph tree not implemented yet.")

    def __get_tree(self, uuid: str, output_id: str) -> BaseResponse[TableTreeResponse]:
        """Get the data tree for the given experiment UUID and output type.

        Args:
            uuid (str): The UUID which the tree belongs to.
            output_id (str): The output ID which the tree belongs to.

        Returns:
            BaseResponse[TableTreeResponse]: The data tree for the given UUID and output type.
        """

        # Get the data tree for the given experiment UUID and output type
        command = ["python3", self.tsp_client_name, "--get-tree", output_id, "--uuid", uuid]
        execution = self.run_tsp_command(command)
        if execution.error or execution.result is None:
            return BaseResponse(error=execution.error)

        process_output = execution.result

        tree = PatternExtractor.extract_tree(process_output)
        return BaseResponse(result=tree)
