import subprocess

from tmll.tsp.utils.path_validator import PathValidator


class TSPService:
    """
    This class is a base class for all TSP services, providing common methods for them.

    Attributes:
        - tsp_client_path (str): The path to the TSP client.
        - tsp_client_name (str): The name of the TSP client script.

    """

    def __init__(self, tsp_client_path: str, tsp_client_name: str) -> None:
        self.tsp_client_path = tsp_client_path
        self.tsp_client_name = tsp_client_name

    def is_tsp_client_path_valid(self) -> bool:
        """
        Check if the TSP client path is valid.
        We can also query a simple -h or --help command to check if the TSP client is working, but this is not implemented here.
        """

        return PathValidator.is_path_valid(self.tsp_client_path)

    def run_tsp_command(self, command: list[str]) -> dict[str, str]:
        """
        Run a TSP command with the given arguments. This method is used by all TSP services.
        """

        # Check if the TSP client path is valid
        if not self.is_tsp_client_path_valid():
            return {"error": "Invalid TSP client path."}

        # Run the TSP command
        process = subprocess.run(
            command, cwd=self.tsp_client_path, capture_output=True, shell=True)

        process_output = process.stdout.decode("utf-8").strip()

        if process.returncode != 0:
            return {"error": f"Error running TSP command. Full error: {process_output}"}

        return {"output": process_output}
