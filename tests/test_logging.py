from __future__ import annotations

from pathlib import Path

import pytest
from modelwarehouse.utils.logging import MWLogger
from .util import TEST_PATH


class TestLogger:
    def test_invalid_args(self):
        with pytest.raises(KeyError):
            MWLogger("some_log_file", None, "invalid")

    def test_filename_format(self):
        logger = MWLogger("some_log_file")
        assert Path(logger.handler.baseFilename).name == "some_log_file.log"
        assert logger.handler.baseFilename.split("/")[6:] == [
            "modelwarehouse",
            "logs",
            "some_log_file.log",
        ]

    def test_custom_logging_path(self):
        logger = MWLogger("some_log_file", TEST_PATH)
        assert logger.handler.baseFilename.split("/")[-3:] == [
            "modelwarehouse",
            "tests",
            "some_log_file.log",
        ]
