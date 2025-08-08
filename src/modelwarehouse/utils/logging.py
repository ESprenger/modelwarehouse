import logging
import os
from pathlib import Path
from typing import Optional

_default_logging_path: Path = (
    Path(os.path.abspath(__file__)).parent.parent.parent / "logs"
)

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
        filepath: Optional[str | Path] = None,
        level: str = "warning",
    ) -> None:
        """Init for Logger object.

        Parameters
        ----------
        filename : str
            name of logging file
        filepath : Optional[str | Path]
            optional filepath (str | Path) to logging file directory. defaults
            to local logging dir
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

    def _define_filepath(self, filename: str, filepath: Optional[str | Path]) -> str:
        filename = f"{filename}" if filename.endswith(".log") else f"{filename}.log"
        filepath = _default_logging_path if not filepath else Path(filepath)
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
