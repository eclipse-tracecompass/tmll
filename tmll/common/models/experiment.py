from typing import List, Optional, Union

from tmll.common.models.trace import Trace
from tmll.common.models.output import Output


class Experiment(Trace):
    """A class to represent an experiment from the TSP server.

    Attributes:
        name (str): The name of the experiment.
        uuid (str): The UUID of the experiment.
        start (int): The start time of the experiment.
        end (int): The end time of the experiment.
        num_events (int): The number of events in the experiment.
        indexing (str): The indexing of the experiment.
        traces (List[Trace]): The list of traces in the experiment.
    """

    def __init__(self, name: str, uuid: str, start: int, end: int, num_events: int, indexing: str, traces: List[Trace] = []) -> None:
        super().__init__(name=name, uuid=uuid, start=start, end=end, num_events=num_events, indexing=indexing, path="")
        self.traces = traces

        self.outputs = []

    @classmethod
    def from_trace(cls, trace_: Trace, traces: List[Trace] = []) -> 'Experiment':
        return cls(trace_.name, trace_.uuid, trace_.start, trace_.end, trace_.num_events, trace_.indexing, traces)

    @classmethod
    def from_tsp_experiment(cls, tsp_experiment) -> 'Experiment':
        traces = [Trace.from_tsp_trace(tsp_trace) for tsp_trace in tsp_experiment.traces.traces]
        return cls(tsp_experiment.name, tsp_experiment.UUID, tsp_experiment.start, tsp_experiment.end, tsp_experiment.number_of_events, tsp_experiment.indexing_status.name, traces)

    def assign_outputs(self, outputs: List[Output]) -> None:
        self.outputs = outputs

    def __repr__(self) -> str:
        return f"Experiment(name={self.name}, uuid={self.uuid}, start={self.start}, end={self.end}, num_events={self.num_events}, indexing={self.indexing}, traces={self.traces}, outputs={self.outputs})"

    def find_outputs(self, keyword: Optional[Union[str, List[str]]] = None, type: Optional[Union[str, List[str]]] = None, match_any: bool = False) -> List[Output]:
        """
        Find outputs based on various criteria with flexible matching logic.
            
        Examples:
            # AND logic (default)
            find_outputs(keyword="cpu", type="time_series")  # Must match both
            
            # OR logic
            find_outputs(keyword="cpu", type="event", match_any=True)  # Can match either
            
            # Multiple values for each criteria
            find_outputs(
                keyword=["cpu", "memory"],     # With match_any=False: must contain both words
                type=["time_graph", "xy"],  # With match_any=True: can contain either word
            )
        
        :param keyword: The keyword(s) to search for in the output name, description, and id
        :type keyword: str or List[str], optional
        :param type: The type(s) to search for in the output type
        :type type: str or List[str], optional
        :param match_any: Whether to match any criteria (OR) or all criteria (AND). Default is False
        :type match_any: bool, optional
        :return: The list of outputs that match the given criteria
        :rtype: List[Output]
        """
        # Convert single strings to lists for consistent processing
        keywords = [keyword] if isinstance(keyword, str) else keyword
        types = [type] if isinstance(type, str) else type
        
        # If no criteria provided, return all outputs
        if not any([keywords, types]):
            return self.outputs

        # Helper function to check if text contains any/all keywords
        def matches_keywords(text: str, keys: List[str], match_any: bool) -> bool:
            text = text.lower()
            keys = [k.lower() for k in keys]

            if not keys:
                return True
            
            if match_any:
                return any(k in text for k in keys)
            return all(k in text for k in keys)

        matches = []
        for output in self.outputs:
            # Check keywords in name, description, and id
            search_text = f"{output.name} {output.description} {output.id}"
            keywords_match = matches_keywords(search_text, keywords or [], match_any)
            
            # Check type
            type_match = matches_keywords(output.type, types or [], match_any)
            
            # Combine all criteria based on match_any
            if match_any:
                # OR: include if any criteria matches
                if keywords_match or type_match:
                    matches.append(output)
            else:
                # AND: include only if all criteria match
                if keywords_match and type_match:
                    matches.append(output)
        
        return sorted(matches, key=lambda x: x.name)