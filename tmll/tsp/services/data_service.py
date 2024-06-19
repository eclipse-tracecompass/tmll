from typing import Dict, List, Optional

from tmll.tsp.services.tsp_service import TSPService


class DataService(TSPService):

    def get_data(self, uuid: str, analysis_type_id: str, data_type: str, **kwargs) -> Dict[object, object]:
        # Get the data for the given experiment UUID, analysis type, and data type
        # If there was an error, return an empty dictionary along with the error message
        if data_type == "xy":
            return self.__get_xy_data(uuid=uuid, analysis_type_id=analysis_type_id, items=kwargs.get("items", []), start_time=kwargs.get("start_time", 0), end_time=kwargs.get("end_time", 0), num_items=kwargs.get("num_items", 100), get_all=kwargs.get("get_all", False))
        elif data_type == "timegraph":
            return self.__get_timegraph_data(uuid=uuid, analysis_type_id=analysis_type_id)
        elif data_type == "table":
            return self.__get_table_data(uuid=uuid, analysis_type_id=analysis_type_id, start_index=kwargs.get("start_index", 0), num_items=kwargs.get("num_items", 100), get_all=kwargs.get("get_all", False))
        else:
            return {"error": "Invalid data type."}

    def __get_xy_data(self, uuid: str, analysis_type_id: str, items: List[int], start_time: int, end_time: int, num_items: Optional[int] = 100, get_all: Optional[bool] = False) -> Dict[object, object]:
        # Get the XY data for the given experiment UUID, analysis type, and other parameters
        # If there was an error, return an empty dictionary along with the error message

        return {}

    def __get_timegraph_data(self, uuid: str, analysis_type_id: str) -> Dict[object, object]:
        # Not implemented yet
        return {"error": "Not implemented yet."}

    def __get_table_data(self, uuid: str, analysis_type_id: str, start_index: int, num_items: Optional[int] = 100, get_all: Optional[bool] = False) -> dict[object, object]:
        # Get the table data for the given experiment UUID, analysis type, and other parameters
        # If there was an error, return an empty dictionary along with the error message

        return {}
