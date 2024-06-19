import re
from typing import Dict, List, Union
from types import NoneType


class PatternExtractor:
    @staticmethod
    def extract_trace_experiment_features(input: str) -> Union[NoneType, Dict[str, Union[str, int]]]:
        pattern = r"(.+?):\s*.*\s*\((.*?)\)\s*start=(\d+)\s*end=(\d+)\s*nbEvents=(\d+)\s*indexing=(\w+)"

        match = re.search(pattern, input.strip())
        if not match:
            return None

        return {
            "name": match.group(1).strip(),
            "uuid": match.group(2).strip(),
            "start": int(match.group(3).strip()),
            "end": int(match.group(4).strip()),
            "num_events": int(match.group(5).strip()),
            "indexing": match.group(6).strip()
        }

    @staticmethod
    def extract_experiments(input: str) -> List[str]:
        pattern = r"Experiment:\s*---+\s*([\s\S]+?)(?=Experiment:|$)"
        matches = re.finditer(pattern, input)

        experiments = [match.group(1) for match in matches]
        return experiments

    @staticmethod
    def extract_output_features(input: str) -> Dict[str, str]:
        input = input.strip()

        name = input.split(":")[0]
        id_match = re.findall(r"\((.*?)\)", input)
        if not id_match:
            return {"error": f"Error extracting ID for analysis type {input}"}

        output_id = id_match[-1]

        return {
            "name": name.strip(),
            "id": output_id.strip()
        }

    @staticmethod
    def extract_xy_tree(input: str) -> Dict[str, Dict[str, int]]:
        pattern = r"^([\w\[\]:.]+)\s+\(([\w\[\]:.]+),\s+(\d+)\)\s+(-?\d+)$"

        result = {}
        for line in input.splitlines():
            # Remove leading special characters until reaching an alphabetical letter
            cleaned_line = re.sub(r"^[^a-zA-Z]+", "", line)

            match = re.match(pattern, cleaned_line)
            if match:
                node_name, _, node_id, parent_id = match.groups()
                result[node_name] = {
                    "id": int(node_id),
                    "parent_id": int(parent_id) if int(parent_id) != -1 else None
                }

        return result
