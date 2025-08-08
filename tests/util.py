from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any

from tests import TEST_PATH


@dataclass
class Case:
    val: Any
    expected_result: Any
    id: str = field(init=False)

    def __post_init__(self):
        self.id = f"case_value - {self.val}"


def clear_filestorage(stub="test_db"):
    for file in glob.glob(f"./test/{stub}.fs*"):
        try:
            os.remove(file)
        except OSError as err:
            print(err)


def gen_test_path(val: str) -> str:
    return os.path.join(TEST_PATH, val)
