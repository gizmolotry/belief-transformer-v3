import pytest
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def shared_pipeline_state():
    """
    Centralized Pytest fixture that instantiates the DAG memory.
    Freezes test-results directory to prevent IO clashes.
    """
    with tempfile.TemporaryDirectory() as td:
        yield {"temp_dir": Path(td), "dag_state": "frozen"}
