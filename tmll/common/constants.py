"""
This module contains the constants used in the project.

Attributes:
"""


class TSP:
    """
    Candidate outputs for the Trace Server Protocol (TSP).
    These are the outputs that are expected to fetch automatically from the TSP server when the user has not specified any outputs.
    In other words, these outputs are kind of most common and useful outputs that the user might want to fetch and use.
    """
    CANDIDATE_OUTPUTS = [
        "org.eclipse.tracecompass.internal.provisional.tmf.core.model.events.TmfEventTableDataProvider"
    ]
