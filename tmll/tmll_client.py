"""
Trace Compass Machine Learning Library (TMLL) is a Python-based library that allows users to apply various machine learning techniques on the analyses of Trace Compass.
The library is implemented as a set of Python classes that can be used to interact with Trace Compass Server Protocol (TSP) and apply machine learning techniques on the data.
"""

class TMLLClient:

    def __init__(self, tsp_server_host: str, tsp_server_port: int) -> None:
        self.tsp_server_host = tsp_server_host
        self.tsp_server_port = tsp_server_port
