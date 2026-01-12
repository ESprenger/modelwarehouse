from __future__ import annotations

import sys
from datetime import datetime
from operator import eq, ge, gt, le, lt
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pytest
from src.modelwarehouse.utils import (
    _json_dumps,
    _resolve_type,
    infer_obj_module,
    resolve_search,
    MWLogger
)

from .util import Case

infer_module_test_cases = [
    Case(Path(), "pathlib"),
    Case(pd.DataFrame(), "pandas"),
    Case(np.array([]), "numpy"),
    Case(str, "builtins"),
    Case(None, "builtins"),
]


class TestInferObjModule:
    @pytest.mark.parametrize("test_case", infer_module_test_cases, ids=lambda tc: tc.id)
    def test_output(self, test_case):
        assert infer_obj_module(test_case.val) == test_case.expected_result


resolve_type_test_cases = [
    Case("5", 5),
    Case("5.46", 5.46),
    Case("something", "something"),
    Case("something else", "something else"),
    Case("user@gmail.com", "user@gmail.com"),
    Case("2020-05-01", pd.to_datetime("2020-05-01")),
    Case("2020-05-01 04:03:01", pd.to_datetime("2020-05-01 04:03:01")),
]


class TestResolveType:
    @pytest.mark.parametrize("test_case", resolve_type_test_cases, ids=lambda tc: tc.id)
    def test_output(self, test_case):
        assert _resolve_type(test_case.val) == test_case.expected_result


resolve_search_test_cases = [
    Case(">5", (gt, 5)),
    Case(">5.46", (gt, 5.46)),
    Case(">some_string", (gt, "some_string")),
    Case(">2020-05-01 04:03:01", (gt, pd.to_datetime("2020-05-01 04:03:01"))),
    Case(">=5", (ge, 5)),
    Case(">=5.46", (ge, 5.46)),
    Case(">=some_string", (ge, "some_string")),
    Case(">=2020-05-01 04:03:01", (ge, pd.to_datetime("2020-05-01 04:03:01"))),
    Case("==5", (eq, 5)),
    Case("==5.46", (eq, 5.46)),
    Case("==some_string", (eq, "some_string")),
    Case("==2020-05-01 04:03:01", (eq, pd.to_datetime("2020-05-01 04:03:01"))),
    Case("<5", (lt, 5)),
    Case("<5.46", (lt, 5.46)),
    Case("<some_string", (lt, "some_string")),
    Case("<2020-05-01 04:03:01", (lt, pd.to_datetime("2020-05-01 04:03:01"))),
    Case("<=5", (le, 5)),
    Case("<=5.46", (le, 5.46)),
    Case("<=some_string", (le, "some_string")),
    Case("<=2020-05-01 04:03:01", (le, pd.to_datetime("2020-05-01 04:03:01"))),
]


class TestResolveSearch:
    @pytest.mark.parametrize(
        "test_case", resolve_search_test_cases, ids=lambda tc: tc.id
    )
    def test_output(self, test_case):
        assert resolve_search(test_case.val) == test_case.expected_result


resolve_json_dump_cases = [
    Case(
        datetime(2025, 1, 1, 4, 4, 4),
        '"{}"'.format(datetime(2025, 1, 1, 4, 4, 4).isoformat(timespec="microseconds")),
    ),
    Case(5, "5"),
    Case(7.46, "7.46"),
    Case(("something", 5, 7.46), '["something", 5, 7.46]'),
    Case(
        pd.DataFrame([[1, 2]], columns=["a", "b"]),
        f"{sys.getsizeof(pd.DataFrame([[1,2]],columns=['a','b']))}",
    ),
]


class TestJsonDumps:
    @pytest.mark.parametrize("test_case", resolve_json_dump_cases, ids=lambda tc: tc.id)
    def test_output(self, test_case):
        assert _json_dumps(test_case.val) == test_case.expected_result

class TestLogger:
    def test_invalid_args(self):
        with pytest.raises(KeyError):
            MWLogger("some_log_file", None, "invalid")

    def test_filename_format(self):
        logger = MWLogger("some_log_file", "./tests/resources")
        assert Path(logger.handler.baseFilename).name == "some_log_file.log"
        assert logger.handler.baseFilename.split("/")[-3:] == [
            "tests",
            "resources",
            "some_log_file.log"
        ]

    def test_custom_logging_path(self):
        TEST_FILEPATH = os.path.join(os.path.dirname(__file__),"resources")
        logger = MWLogger("some_log_file", TEST_FILEPATH)
        assert logger.handler.baseFilename.split("/")[-3:] == [
            "tests",
            "resources",
            "some_log_file.log"
        ]
