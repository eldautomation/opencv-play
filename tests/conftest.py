import os

# load_main requires a real device key (R6). Use an obviously fake one for tests
# unless the environment already provides one.
os.environ.setdefault("DEVICE_ENCRYPTION_KEY", "test-dummy-key")


def pytest_addoption(parser):
    parser.addoption(
        "--keep-output",
        action="store_true",
        default=False,
        help="Keep integration-test output under outputs/ instead of temp dirs.",
    )


def pytest_configure(config):
    import pytest as _pytest
    _pytest.keep_output_flag = config.getoption("--keep-output")
