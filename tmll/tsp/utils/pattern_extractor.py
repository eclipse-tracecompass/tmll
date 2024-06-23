import re
from typing import List, Union
from types import NoneType

from tmll.tsp.models.response.base import BaseResponse
from tmll.tsp.models.response.data.xy import XYDataResponse
from tmll.tsp.models.response.output import OutputResponse
from tmll.tsp.models.response.trace import TraceResponse
from tmll.tsp.models.response.data.xy import XYDataResponse
from tmll.tsp.models.response.tree.node import NodeTreeResponse
from tmll.tsp.models.response.tree.table import TableTreeResponse


class PatternExtractor:
    @staticmethod
    def extract_trace_experiment_features(input: str) -> Union[NoneType, TraceResponse]:
        """Extract the features of a trace experiment (name, UUID, start, end, number of events, indexing)

        Args:
            input (str): The input string to extract the features from. It can be either a trace or an experiment line.

        Returns:
            Union[NoneType, TraceResponse]: The extracted trace experiment features. If there is an error, it returns None.
        """

        pattern = r"(.+?):\s*.*\s*\((.*?)\)\s*start=(\d+)\s*end=(\d+)\s*nbEvents=(\d+)\s*indexing=(\w+)"

        match = re.search(pattern, input.strip())
        if not match:
            return None

        return TraceResponse(
            name=match.group(1).strip(),
            uuid=match.group(2).strip(),
            start=int(match.group(3).strip()),
            end=int(match.group(4).strip()),
            num_events=int(match.group(5).strip()),
            indexing=match.group(6).strip()
        )

    @staticmethod
    def extract_experiments(input: str) -> List[str]:
        """Extract the experiments from the input string.

        Args:
            input (str): The input string to extract the experiments from.

        Returns:
            List[str]: The list of experiments along with their all information.
        """

        pattern = r"Experiment:\s*---+\s*([\s\S]+?)(?=Experiment:|$)"
        matches = re.finditer(pattern, input)

        experiments = [match.group(1) for match in matches]
        return experiments

    @staticmethod
    def extract_output_features(input: str) -> Union[NoneType, OutputResponse]:
        """Extract the output features from the input string, which contains the output type name and ID.

        Args:
            input (str): The input string to extract the output features from.

        Returns:
            Union[NoneType, OutputResponse]: The extracted output features. If there is an error, it returns None.
        """

        input = input.strip()

        name = input.split(":")[0]
        id_match = re.findall(r"\((.*?)\)", input)
        if not id_match:
            return None

        output_id = id_match[-1]

        return OutputResponse(name=name.strip(), id=output_id.strip())

    @staticmethod
    def extract_tree(input: str) -> TableTreeResponse:
        """Extract the tree from the input string, which contains the node name, ID, and parent ID.

        Args:
            input (str): The input string to extract the tree from.

        Returns:
            TableTreeResponse: The extracted tree.
        """

        pattern = r"^([\w\[\]:.]+)\s+\(([\w\[\]:.]+),\s+(\d+)\)\s+(-?\d+)$"

        nodes = []
        for line in input.splitlines():
            # Remove leading special characters until reaching an alphabetical letter
            cleaned_line = re.sub(r"^[^a-zA-Z]+", "", line)

            match = re.match(pattern, cleaned_line)
            if match:
                node_name, _, node_id, parent_id = match.groups()
                nodes.append(NodeTreeResponse(name=node_name, id=int(node_id), parent_id=int(parent_id)))

        return TableTreeResponse(nodes=nodes)

    @staticmethod
    def extract_xy_values(input: str) -> BaseResponse[XYDataResponse]:
        """Extract the X and Y values from the input string.

        Args:
            input (str): The input string to extract the X and Y values from.

        Returns:
            BaseResponse[XYDataResponse]: The extracted X and Y values.
        """

        try:
            x_values = re.findall(r"Series X-values: \[(.*)\]", input)[0].split(',')
            y_values = re.findall(r"Series Y-values: \[(.*)\]", input)[0].split(',')

            return BaseResponse(result=XYDataResponse(x_values=x_values, y_values=y_values))
        except Exception as e:
            return BaseResponse(result=XYDataResponse(x_values=[], y_values=[]))
