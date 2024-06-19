from tmll.tsp.services.tsp_service import TSPService


class TreeService(TSPService):

    def get_tree(self, uuid: str, analysis_type_id: str, tree_type: str, **kwargs) -> dict[object, object]:
        # Get the tree for the given experiment UUID and analysis type
        # If there was an error, return an empty dictionary along with the error message
        if tree_type == "xy":
            return self.__get_tree_xy(uuid, analysis_type_id, **kwargs)
        elif tree_type == "timegraph":
            return self.__get_tree_timegraph(uuid, analysis_type_id, **kwargs)
        elif tree_type == "table":
            return self.__get_tree_table(uuid, analysis_type_id, **kwargs)
        else:
            return {"error": "Invalid tree type."}

    def __get_tree_xy(self, uuid: str, analysis_type_id: str) -> dict[object, object]:
        # Get the XY tree for the given experiment UUID and analysis type
        # If there was an error, return an empty dictionary along with the error message

        return {}

    def __get_tree_timegraph(self, uuid: str, analysis_type_id: str) -> dict[object, object]:
        # Get the timegraph tree for the given experiment UUID and analysis type
        # If there was an error, return an empty dictionary along with the error message

        return {}

    def __get_tree_table(self, uuid: str, analysis_type_id: str) -> dict[object, object]:
        # Get the table tree for the given experiment UUID and analysis type
        # If there was an error, return an empty dictionary along with the error message

        return {}
