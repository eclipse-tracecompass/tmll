import subprocess

from tmll.tsp_legacy.models.response.base import BaseResponse
from tmll.tsp_legacy.utils.path_validator import PathValidator


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
        """This method checks if the TSP client path is valid.

        Returns:
            bool: True if the TSP client path is valid, False otherwise.
        """

        return PathValidator.is_path_valid(self.tsp_client_path)

    def run_tsp_command(self, command: list[str]) -> BaseResponse[str]:
        """This method runs a TSP command in the command line/terminal.

        Args:
            command (list[str]): The command to be run as a list of strings.

        Returns:
            BaseResponse[str]: The result of the command if successful, or an error message if not.
        """

        # Check if the TSP client path is valid
        if not self.is_tsp_client_path_valid():
            return BaseResponse(error="Invalid TSP client path.")

        # Run the TSP command
        process = subprocess.run(command, cwd=self.tsp_client_path, capture_output=True, shell=True)

        process_output = process.stdout.decode("utf-8").strip()

        if process.returncode != 0:
            return BaseResponse(error=f"Error running TSP command. Full error: {process_output}")

        return BaseResponse(result=process_output)
