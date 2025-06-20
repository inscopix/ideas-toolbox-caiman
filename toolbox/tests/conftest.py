import os
import pytest
import shutil


@pytest.fixture
def output_dir():
    """Construct a path for the directory where outputs can be stored,
    and cleans up outputs after each test finishes.
    """
    _output_dir = "/ideas/outputs"
    os.makedirs(_output_dir, exist_ok=True)
    yield _output_dir

    # clean up output dir or else subsequent test runs of MSR will fail
    # because of pre-existing registered cellsets and eventsets
    shutil.rmtree(_output_dir)
