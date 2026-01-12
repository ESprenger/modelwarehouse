import hashlib
import json
import logging
import re
import sys
from ast import literal_eval
from ctypes import c_int32
from datetime import datetime
from inspect import getmodule
from operator import eq, ge, gt, le, lt
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union


from pandas import Timestamp, to_datetime

_op_lookup: Dict[str, Callable] = {">": gt, "<": lt, ">=": ge, "<=": le, "==": eq}

_level_dict: dict = {
    "notset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class MWLogger:
    """
    A class for core logging for ModelWarehouse.

    ...

    Attributes
    ----------
    logger : logging.Logger
        core logging object
    handler : logging.FileHandler
        file oriented logging handler

    """

    def __init__(
        self,
        filename: str,
        filepath: Union[str, Path],
        level: str = "warning",
    ) -> None:
        """Init for Logger object.

        Parameters
        ----------
        filename : str
            name of logging file
        filepath : Path
            filepath (str | Path) to logging file directory
        level : str
            set max debugging level. defaults to "warning"

        """

        logging.basicConfig(
            level=self._return_level(level),
            format="%(asctime)s - %(levelname)s : %(message)s",
        )

        self.logger = logging.getLogger(__name__)
        self.logger.handlers.clear()
        self.handler = logging.FileHandler(
            self._define_filepath(filename, filepath), mode="a"
        )
        self.handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s : %(message)s")
        )
        self.logger.addHandler(self.handler)

    def _return_level(self, level: str):
        try:
            return _level_dict[level.lower()]
        except Exception as _:
            raise KeyError(
                "Valid inputs for 'level' - notset,debug,info,warning,error,critical !"
            )

    def _define_filepath(self, filename: str, filepath: Union[str,Path]) -> str:
        filename = f"{filename}" if filename.endswith(".log") else f"{filename}.log"
        filepath = Path(filepath)
        return str(filepath / filename)

    @property
    def append(self):
        """Return logging.Logger object

        Examples
        --------

        myLogger.append.error(...)
        myLogger.append.debug(...)

        """

        return self.logger


def infer_obj_module(model_object: Any) -> str:
    """Return information on associated module of object.

    Parameters
    ----------
    model_object : Any
        A ML model object.

    Returns
    -------
    str
        formatted info string

    """
    mod_search = re.compile(r"\<(.*?)\>")
    type_search = re.compile(r"'(.*?)'")
    mod_match = mod_search.search(str(getmodule(model_object)))
    type_match = type_search.findall(str(type(model_object)))[0].split(".")

    if mod_match:
        return (mod_match.group(1).split(" ")[1]).replace("'", "").split(".")[0]

    return type_match[0] if len(type_match) > 1 else "builtins"


def _resolve_type(val: str) -> Any:
    """Typecasting string to its literal type.

    Parameters
    ----------
    val : str
        input value

    Returns
    -------
    Any
        Any literal type.

    """

    for fn in [literal_eval, to_datetime]:
        try:
            return fn(val)
        except Exception as _:
            continue
    return val


def resolve_search(val: Any) -> Tuple[Callable, Any]:
    """Given string search expression of format "<comparison operator><value>", converts into tuple of comparison method + literal value.

    Parameters
    ----------
    val : Any
        formatted filter expression

    Returns
    -------
    Tuple[Callable, Any]
        tuple of comparison operator (e.g. >) and literal value

    Examples
    --------
    ">=5.46" --> (>= , 5.46) (callable,int)

    """

    cmp, val = re.findall(r"""(?P<cmp>[<>=]+)'?(?P<value>.*)'?""", val)[0]
    return _op_lookup[cmp], _resolve_type(val)


def _json_default(val: Any) -> Union[Tuple, Any]:
    """Convert literals to types that can be processed by Json.

    Parameters
    ----------
    val : Any
        Any literal type.

    Returns
    -------
    Union[Tuple, Any]
        Tuple of converted values or single converted value.

    """

    if isinstance(val, tuple):
        return tuple(map(_json_default, val))
    elif isinstance(val, str | int | float):
        return val
    elif isinstance(val, Timestamp | datetime):
        return val.isoformat(timespec="microseconds")
    else:
        return sys.getsizeof(val)


def _json_dumps(val: Tuple | Any) -> str:
    """Convert value into JSON formatted string.

    Parameters
    ----------
    val : Tuple | Any
        Either Tuple of literal(s) or single literal.

    Returns
    -------
    str
        Formatted string.

    """
    return json.dumps(val, default=_json_default)


def produce_hash(val: Tuple | Any) -> int:
    """Converts value into JSON formatted string, then converts to md5 hash,
       and then finally converts to C type 32 bit integer.  Used as a
       deterministic hashing function that is consistent across python
       instances.

    Parameters
    ----------
    val : Tuple | Any
        Tuple of any literal(s) or single literal.

    Returns
    -------
    int
        32 bit Integer

    """
    val = val if isinstance(val, tuple) else (val,)
    return c_int32(
        int(hashlib.md5(_json_dumps(val).encode("utf-8")).digest().hex(), 16)
    ).value
