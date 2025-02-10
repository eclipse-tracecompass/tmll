import os
from typing import Dict, List
import pandas as pd
import pytest
from tmll.common.models.output import Output
from tmll.common.models.tree.tree import Tree
from tmll.tmll_client import TMLLClient
from tmll.tsp.tsp.indexing_status import IndexingStatus

# Constants
TSP_HOST = os.getenv('TEST_TSP_HOST', 'localhost')
TSP_PORT = int(os.getenv('TEST_TSP_PORT', '8081'))
EXPERIMENT_NAME = "Test Experiment"
OUTPUTS = ["CPU Usage", "Disk I/O View", "Histogram", "Events Table", "Resources Status"]


@pytest.fixture
def kernel():
    return os.path.join(os.path.dirname(__file__), "test-trace-files", "ctf", "src", "main", "resources", "kernel")


@pytest.fixture
def ust():
    return os.path.join(os.path.dirname(__file__), "test-trace-files", "ctf", "src", "main", "resources", "context-switches", "context-switches-ust")


class TestTMLLClient:
    """Test suite for the TMLLClient class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        pass

    def _init_client(self, host: str = TSP_HOST, port: int = TSP_PORT) -> TMLLClient:
        """Helper method to initialize TMLLClient."""
        return TMLLClient(tsp_server_host=host, tsp_server_port=port, delete_all=True)

    def _setup_traces(self, traces: List[str]) -> List[Dict[str, str]]:
        """Helper method to setup traces for experiment creation."""
        return [{"path": trace} for trace in traces]

    @pytest.mark.parametrize("port,expected_status", [(TSP_PORT, 200), (TSP_PORT + 1, 400)])
    def test_client_initialization(self, port: int, expected_status: int):
        """Test client initialization with various health status codes."""
        client = self._init_client(port=port)
        assert client.health_status == expected_status, f"Expected status {expected_status}, got {client.health_status}"

    def test_create_experiment_single(self, kernel):
        """Test successful experiment creation with valid traces."""
        client = self._init_client()
        traces = self._setup_traces([kernel])
        experiment = client.create_experiment(traces=traces, experiment_name=EXPERIMENT_NAME)

        # Verify experiment creation
        assert experiment is not None, "Experiment should not be None"
        assert experiment.name == EXPERIMENT_NAME, f"Experiment name should match ({experiment.name} != {EXPERIMENT_NAME})"
        assert len(experiment.traces) == 1, "Expected 1 trace in experiment"
        assert experiment.indexing == IndexingStatus.COMPLETED.name  # Indexing should be completed

        # Verify experiment"s outputs (some surely expected outputs)
        experiment_outputs = experiment.find_outputs(keyword=OUTPUTS, match_any=True)
        assert len(experiment_outputs) == len(OUTPUTS), f"Expected {len(OUTPUTS)} outputs, got {len(experiment_outputs)}"
        assert all(output.name in OUTPUTS for output in experiment_outputs), "All outputs should be in expected list"

    def test_create_experiment_multiple(self, kernel, ust):
        """Test successful experiment creation with multiple valid traces."""
        client = self._init_client()
        experiment = client.create_experiment(traces=self._setup_traces([kernel, ust]), experiment_name=EXPERIMENT_NAME)

        # Verify experiment creation
        assert experiment is not None, "Experiment should not be None"
        assert experiment.name == EXPERIMENT_NAME, f"Experiment name should match ({experiment.name} != {EXPERIMENT_NAME})"
        assert len(experiment.traces) == 2, "Expected 2 traces in experiment"
        assert experiment.indexing == IndexingStatus.COMPLETED.name

    def test_create_experiment_corner_cases(self):
        """Test experiment creation corner cases."""
        client = self._init_client()

        # Test with empty traces
        assert client.create_experiment([], EXPERIMENT_NAME) is None, "Experiment should be None with empty traces"

        # Test with failed trace opening
        try:
            traces = self._setup_traces(["/invalid/path"])
            assert client.create_experiment(traces, EXPERIMENT_NAME) is None, "Experiment should be None with invalid path"
        except Exception:
            pass  # The client would raise an exception in this case, which is expected

    def test_fetch_outputs_with_tree(self, kernel, ust):
        """Test fetching outputs with tree data."""
        client = self._init_client()
        experiment = client.create_experiment(traces=self._setup_traces([kernel, ust]), experiment_name=EXPERIMENT_NAME)
        assert experiment is not None, "Experiment should not be None"

        outputs = experiment.find_outputs(keyword=OUTPUTS, match_any=True)
        result = client.fetch_outputs_with_tree(experiment, [output.id for output in outputs])

        assert result is not None, "Outputs and tree data should not be None"
        assert isinstance(result, list), "Response should be a list"
        assert len(result) == len(OUTPUTS) == len(outputs), f"Expected number of outputs and tree data ({len(OUTPUTS)}), got {len(result)}"
        assert all(isinstance(res["output"], Output) for res in result), "Output data should be of type Output"
        assert all(isinstance(res["tree"], Tree) for res in result), "Tree data should be of type Tree"
        assert all(output.id in [res["output"].id for res in result] for output in outputs), "All output IDs should be in response"  # type: ignore
        assert all(res["tree"] is not None for res in result), "Tree data should not be None"
        assert all(len(res["tree"].nodes) > 0 for res in result), "Tree data should have nodes"  # type: ignore

    def test_fetch_xy_output(self, kernel):
        """Test XY output type data fetching."""
        client = self._init_client()
        experiment = client.create_experiment(traces=self._setup_traces([kernel]), experiment_name=EXPERIMENT_NAME)
        assert experiment is not None, "Experiment should not be None"
        assert experiment.indexing == IndexingStatus.COMPLETED.name, "Experiment should be indexed as COMPLETED"
        assert experiment.outputs is not None, "Experiment should have outputs"

        # Only fetch CPU Usage as XY output
        ot = client.fetch_outputs_with_tree(experiment, [output.id for output in experiment.outputs if output.name == "CPU Usage"])
        assert ot is not None, "Output and tree data should not be None"
        assert len(ot) == 1, "Expected 1 output and tree data (CPU Usage)"
        assert ot[0]["output"] is not None, "Output data should not be None"
        assert ot[0]["tree"] is not None, "Tree data should not be None"
        assert isinstance(ot[0]["output"], Output), "Output data should be of type Output"
        assert isinstance(ot[0]["tree"], Tree), "Tree data should be of type Tree"

        xy = client.fetch_data(experiment, ot)
        cpu_id = ot[0]["output"].id
        assert xy and isinstance(xy, dict), "Response should be non-empty dictionary"
        assert len(xy) == 1, f"Expected 1 entry, got {len(xy)}"
        assert cpu_id in xy, f"Missing CPU ID {cpu_id} in response"
        assert isinstance(xy[cpu_id], dict), "CPU data should be dictionary"
        for name, df in xy[cpu_id].items():
            assert isinstance(df, pd.DataFrame), f"{name}: Expected DataFrame"
            assert df.columns.tolist() == ["x", "y"], f"{name}: Expected columns 'x' and 'y'"
            assert df["y"].dtype == float, f"{name}: 'y' column should contain floats"

    def test_fetch_table_output(self, kernel):
        """Test Table output type data fetching."""
        client = self._init_client()
        experiment = client.create_experiment(traces=self._setup_traces([kernel]), experiment_name=EXPERIMENT_NAME)
        assert experiment is not None, "Experiment should not be None"
        assert experiment.indexing == IndexingStatus.COMPLETED.name, "Experiment should be indexed as COMPLETED"
        assert experiment.outputs is not None, "Experiment should have outputs"

        # Only fetch Events Table as Table output
        ot = client.fetch_outputs_with_tree(experiment, [output.id for output in experiment.outputs if output.name == "Events Table"])
        assert ot is not None, "Output and tree data should not be None"
        assert len(ot) == 1, "Expected 1 output and tree data (Events Table)"
        assert ot[0]["output"] is not None, "Output data should not be None"
        assert ot[0]["tree"] is not None, "Tree data should not be None"
        assert isinstance(ot[0]["output"], Output), "Output data should be of type Output"
        assert isinstance(ot[0]["tree"], Tree), "Tree data should be of type Tree"

        table = client.fetch_data(experiment, ot)
        events_id = ot[0]["output"].id
        assert table and isinstance(table, dict), "Response should be non-empty dictionary"
        assert len(table) == 1, f"Expected 1 entry, got {len(table)}"
        assert events_id in table, f"Missing Events Table ID {events_id} in response"

        events_table = table[events_id]
        assert isinstance(events_table, pd.DataFrame), "Events Table data should be DataFrame"
        assert not events_table.empty, "Events Table should not be empty"
        assert all(col in events_table.columns for col in ["Timestamp", "Event type", "Contents"]), "Expected columns in Events Table"

    def test_fetch_timegraph_output(self, kernel):
        """Test Timegraph output type data fetching."""
        client = self._init_client()
        experiment = client.create_experiment(traces=self._setup_traces([kernel]), experiment_name=EXPERIMENT_NAME)
        assert experiment is not None, "Experiment should not be None"
        assert experiment.indexing == IndexingStatus.COMPLETED.name, "Experiment should be indexed as COMPLETED"
        assert experiment.outputs is not None, "Experiment should have outputs"

        # Only fetch Resources Status as Timegraph output
        ot = client.fetch_outputs_with_tree(experiment, [output.id for output in experiment.outputs if output.name == "Resources Status"])
        assert ot is not None, "Output and tree data should not be None"
        assert len(ot) == 1, "Expected 1 output and tree data (Resources Status)"
        assert ot[0]["output"] is not None, "Output data should not be None"
        assert ot[0]["tree"] is not None, "Tree data should not be None"
        assert isinstance(ot[0]["output"], Output), "Output data should be of type Output"
        assert isinstance(ot[0]["tree"], Tree), "Tree data should be of type Tree"

        timegraph = client.fetch_data(experiment, ot)
        res_id = ot[0]["output"].id
        assert timegraph and isinstance(timegraph, dict), "Response should be non-empty dictionary"
        assert len(timegraph) == 1, f"Expected 1 entry, got {len(timegraph)}"
        assert res_id in timegraph, f"Missing Resources Status ID {res_id} in response"

        resources = timegraph[res_id]
        assert isinstance(resources, pd.DataFrame), "Resources Status data should be DataFrame"
        assert not resources.empty, "Resources Status should not be empty"
        assert all(col in resources.columns for col in ["label", "start_time", "end_time", "entry_name", "entry_id"]), "Expected columns in Resources Status"
