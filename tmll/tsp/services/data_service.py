from typing import Any, Dict, List, Optional, Union

from tmll.tsp.services.tsp_service import TSPService
from tmll.tsp.utils.pattern_extractor import PatternExtractor

AVAILABLE_DATA_TYPES = ["xy", "timegraph", "table"]


class DataService(TSPService):

    def get_data(self, uuid: str, output_id: str, data_type: str, **kwargs) -> Union[Dict[str, str], Dict[Any, Any]]:
        """Get the data for the given UUID, output, and data type.

        Args:
            uuid (str): The UUID to get the data from.
            output_id (str): The output ID to get the data from.
            data_type (str): The type of data to get. You can use get_data_types() to get the available data types.

        Returns:
            Union[Dict[str, str], Dict[Any, Any]]: The data for the given UUID, output, and data type.
        """

        # Get the data for the given UUID, output, and data type
        if data_type == "xy":
            return self.__get_xy_data(uuid=uuid, output_id=output_id, items=kwargs.get("items", []),
                                      start_time=kwargs.get("start_time", 0), end_time=kwargs.get("end_time", 0),
                                      num_items=kwargs.get("num_items", 100), get_all=kwargs.get("get_all", False))
        elif data_type == "timegraph":
            return self.__get_timegraph_data(uuid=uuid, output_id=output_id)
        elif data_type == "table":
            return self.__get_table_data(uuid=uuid, output_id=output_id, start_index=kwargs.get("start_index", 0),
                                         num_items=kwargs.get("num_items", 100), get_all=kwargs.get("get_all", False))
        else:
            return {"error": f"Invalid data type. Available data types: {', '.join(AVAILABLE_DATA_TYPES)}"}

    @staticmethod
    def get_data_types() -> List[str]:
        """Get the available data types.

        Returns:
            List[str]: The available data types.
        """
        return AVAILABLE_DATA_TYPES

    def __get_xy_data(self, uuid: str, output_id: str, items: List[int], start_time: int, end_time: int, num_items: Optional[int] = 100, get_all: Optional[bool] = False) -> Union[Dict[str, str], Dict[str, List[Union[int, float]]]]:
        """Get the XY data for the given UUID, output, and other parameters.

        Args:
            uuid (str): The UUID to get the data from.
            output_id (str): The output ID to get the data from.
            items (List[int]): Which items to get the XY data for (i.e., derieved from the output's tree structure)
            start_time (int): The start time to get the XY data for.
            end_time (int): The end time to get the XY data for.
            num_items (Optional[int], optional): The number of items to fetch the data in the given time range. Defaults to 100.
            get_all (Optional[bool], optional): If True, it will get all of the data points in the given time range. Defaults to False.

        Returns:
            Union[Dict[str, str], Dict[str, List[Union[int, float]]]]: The XY data for the given UUID, output, and other parameters.
        """

        # Initialize the output values
        output_x_values, output_y_values = [], []

        while True:
            command = [
                'python', 'tsp_cli_client', '--get-xy', output_id,
                '--uuid', uuid, '--items', ' '.join(map(str, items)),
                '--time-range', str(start_time), str(end_time), str(num_items)
            ]
            execution = self.run_tsp_command(command)

            if "error" in execution:
                return execution

            process_output = execution.get("output", "")

            # Extract the XY values from the process output
            x_values, y_values = PatternExtractor.extract_xy_values(process_output)

            # If there are no XY values, break the loop (i.e., no more data to fetch)
            if not x_values or not y_values or (len(x_values) == 1 and x_values[0] == ''):
                break

            # Append the XY values to the output lists
            output_x_values.extend(map(int, x_values))
            output_y_values.extend(map(float, y_values))

            # If not get_all, break the loop (i.e., just one-time data fetch)
            if not get_all:
                break

            # Update the start time for the next iteration
            start_time = output_x_values[-1] + 1

        return {"x": output_x_values, "y": output_y_values}

    def __get_timegraph_data(self, uuid: str, output_id: str) -> Dict[str, str]:
        # Not implemented yet
        return {"error": "Not implemented yet."}

    def __get_table_data(self, uuid: str, output_id: str, start_index: int, num_items: Optional[int] = 100, column_ids: List[int] = [], get_all: Optional[bool] = False) -> Union[Dict[str, str], Dict[int, List[str]]]:
        """Get the table data for the given UUID, output, and other parameters.

        Args:
            uuid (str): The UUID to get the data from.
            output_id (str): The output ID to get the data from.
            start_index (int): The start index to get the table data for.
            num_items (Optional[int], optional): The number of rows to get the data from the table. Defaults to 100.
            column_ids (List[int], optional): The specific column ids to get the data from the table. Defaults to [].
            get_all (Optional[bool], optional): If True, it will get all of the data points, starting from the start index until the end of the table. Defaults to False.

        Returns:
            Union[Dict[str, str], Dict[int, List[str]]]: The table data for the given UUID, output, and other parameters.
        """

        # Initialize the virtual table
        virtual_table = {}

        while True:
            command = [
                'python', 'tsp_cli_client', '--get-virtual-table-lines', output_id,
                '--uuid', uuid, '--table-line-index', str(start_index),
                '--table-line-count', str(num_items)
            ]
            # Add the column ids if provided
            if column_ids:
                command.extend(['--table-column-ids'] + list(map(str, column_ids)))

            execution = self.run_tsp_command(command)

            if "error" in execution:
                return execution

            process_output = execution.get("output", "")

            # Extract the virtual table from the process output
            line_index = -1
            for line in process_output.splitlines():
                line = line.strip()
                if line.startswith('index:'):
                    line_index = int(line.split(':')[1].strip())
                    virtual_table[line_index] = []
                elif line.startswith('"content":'):
                    content = line[10:].strip().replace('\"', '')
                    virtual_table[line_index].append(content)

            # If there are no more lines to fetch, break the loop
            # If not get_all, break the loop (i.e., just one-time data fetch)
            if line_index == -1 or not get_all:
                break

            # Update the start index for the next iteration
            start_index = line_index + 1

        return virtual_table
