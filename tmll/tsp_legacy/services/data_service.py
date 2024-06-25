from typing import List

from tmll.tsp_legacy.models.response.base import BaseResponse
from tmll.tsp_legacy.models.response.data.table.column import TableDataColumnResponse
from tmll.tsp_legacy.models.response.data.table.table import TableDataResponse, TableDataRowResponse
from tmll.tsp_legacy.models.response.data.xy import XYDataResponse
from tmll.tsp_legacy.services.tsp_service import TSPService
from tmll.tsp_legacy.utils.pattern_extractor import PatternExtractor

AVAILABLE_DATA_TYPES = ["xy", "timegraph", "table"]


class DataService(TSPService):

    def get_data(self, uuid: str, output_id: str, data_type: str, **kwargs) -> BaseResponse:
        """Get the data for the given UUID, output, and data type.

        Args:
            uuid (str): The UUID to get the data from.
            output_id (str): The output ID to get the data from.
            data_type (str): The type of data to get. You can use get_data_types() to get the available data types.

        Returns:
            BaseResponse: The data for the given UUID, output, and data type.
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
            return BaseResponse(error=f"Invalid data type. Available data types: {', '.join(AVAILABLE_DATA_TYPES)}")

    def get_table_columns(self, uuid: str, output_id: str) -> BaseResponse[List[TableDataColumnResponse]]:
        """Get the table columns for the given UUID and output ID.

        Args:
            uuid (str): The UUID to get the table columns from.
            output_id (str): The output ID to get the table columns from.

        Returns:
            BaseResponse: The table columns for the given UUID and output ID.
        """

        command = ['python', 'tsp_cli_client', '--get-virtual-table-columns', output_id, '--uuid', uuid]
        execution = self.run_tsp_command(command)

        if execution.error or not execution.result:
            return BaseResponse(error=execution.error)

        process_output = execution.result

        # Extract the table columns from the process output
        columns = PatternExtractor.extract_table_columns(process_output)
        if columns.error or not columns.result:
            return BaseResponse(error=columns.error)

        return columns

    @staticmethod
    def get_data_types() -> List[str]:
        """Get the available data types.

        Returns:
            List[str]: The available data types.
        """
        return AVAILABLE_DATA_TYPES

    def __get_xy_data(self, uuid: str, output_id: str, items: List[int], start_time: int, end_time: int, num_items: int = 100, get_all: bool = False) -> BaseResponse[XYDataResponse]:
        """Get the XY data for the given UUID, output, and other parameters.

        Args:
            uuid (str): The UUID to get the data from.
            output_id (str): The output ID to get the data from.
            items (List[int]): Which items to get the XY data for (i.e., derieved from the output's tree structure)
            start_time (int): The start time to get the XY data for.
            end_time (int): The end time to get the XY data for.
            num_items (int, optional): The number of items to fetch the data in the given time range. Defaults to 100.
            get_all (bool, optional): If True, it will get all of the data points in the given time range. Defaults to False.

        Returns:
            BaseResponse[XYResponse]: The XY data for the given UUID, output, and other parameters.
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

            if execution.error or not execution.result:
                return BaseResponse(error=execution.error)

            process_output = execution.result

            # Extract the XY values from the process output
            xy = PatternExtractor.extract_xy_values(process_output)
            if xy.error or not xy.result:
                return BaseResponse(error=xy.error)

            x_values = xy.result.x_values
            y_values = xy.result.y_values

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

        return BaseResponse(result=XYDataResponse(x_values=output_x_values, y_values=output_y_values))

    def __get_timegraph_data(self, uuid: str, output_id: str) -> BaseResponse:
        """(NOT IMPLEMENTED YET!) Get the timegraph data for the given UUID and output ID.

        Args:
            uuid (str): The UUID to get the data from.
            output_id (str): The output ID to get the data from.

        Returns:
            BaseResponse: The timegraph data for the given UUID and output ID.
        """

        return BaseResponse(error="Timegraph data is not implemented yet.")

    def __get_table_data(self, uuid: str, output_id: str, start_index: int, num_items: int = 100, column_ids: List[int] = [], get_all: bool = False) -> BaseResponse[TableDataResponse]:
        """Get the table data for the given UUID, output, and other parameters.

        Args:
            uuid (str): The UUID to get the data from.
            output_id (str): The output ID to get the data from.
            start_index (int): The start index to get the table data for.
            num_items (int, optional): The number of rows to get the data from the table. Defaults to 100.
            column_ids (List[int], optional): The specific column ids to get the data from the table. Defaults to [].
            get_all (bool, optional): If True, it will get all of the data points, starting from the start index until the end of the table. Defaults to False.

        Returns:
            BaseResponse[TableResponse]: The table data for the given UUID, output, and other parameters.
        """

        # Initialize the virtual table
        virtual_table = []

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
            if execution.error or not execution.result:
                return BaseResponse(error=execution.error)

            process_output = execution.result

            # Extract the virtual table from the process output
            line_index = -1
            values = []
            for line in process_output.splitlines():
                line = line.strip()
                if line.startswith('index:'):
                    line_index = int(line.split(':')[1].strip())
                elif line.startswith('"content":'):
                    content = line[10:].strip().replace('\"', '')
                    values.append(content)

            # If there are no more lines to fetch, break the loop
            if line_index == -1:
                break

            # Append the values to the virtual table
            if values:
                virtual_table.append(TableDataRowResponse(index=line_index, values=values))

            # If not get_all, break the loop (i.e., just one-time data fetch)
            if not get_all:
                break

            # Update the start index for the next iteration
            start_index = line_index + 1

        return BaseResponse(result=TableDataResponse(columns=[], rows=virtual_table))
