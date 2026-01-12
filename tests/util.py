from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Case:
    val: Any
    expected_result: Any
    id: str = field(init=False)

    def __post_init__(self):
        self.id = f"case_value - {self.val}"


def clear_filestorage(stub="test_db"):
    for file in glob.glob(f"./tests/resources/{stub}.fs*"):
        try:
            os.remove(file)
        except OSError as err:
            print(err)
