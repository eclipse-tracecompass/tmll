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
