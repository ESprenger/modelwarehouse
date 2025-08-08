import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

from persistent.mapping import PersistentMapping
import relstorage
import transaction
from modelwarehouse.utils.logging import MWLogger
from ZODB import DB, Connection, FileStorage, config


class ConnectionManager:
    """
    A class for establishing ZODB database connection.

    ...

    Attributes
    ----------
    _config_path : Path
         path to configuration/filestorage
    _storage_path : Optional[FileStorage.FileStorage]
         if using filestorage, path to filestorage file
    _db : MWLogger
         zodb database
    _conn : MWLogger
         database connection
    _root : MWLogger
         root object of database

    Notes
    -----
    This class handles core database functionality - connect, close, commit, cancel etc.
    Any read/write functionality should exist elsewhere, yet use this object as the datbase
    'connection.

    """

    _config_path: Path
    _storage: Optional[FileStorage.FileStorage]
    _db: Optional[DB]
    _conn: Optional[Connection.Connection]
    _root: Optional[PersistentMapping]

    def __init__(
        self,
        path_to_configuration: Union[str, Path],
        log_filename: str,
        log_filepath: Optional[str | Path],
        immediate_init: bool = True,
    ) -> None:
        """Init method

        Parameters
        ----------
        path_to_configuration : Union[str, Path]
            path to databse connection/file storage
        log_filename : str
            name of log file
        log_filepath : Optional[str | Path]
            path to log file
        immediate_init : bool
            bool to init database connection on initalization

        Examples
        --------
        FIXME: Add docs.

        """

        self.logger = MWLogger(
            filename=log_filename, filepath=log_filepath, level="warning"
        )
        self._config_path = Path(path_to_configuration)
        self._storage = None
        self._db = None
        self._conn = None
        self._root = None
        if immediate_init:
            self.create_db_connection()

    def create_db_connection(self):
        """setup database connection

        Raises
        ------

            general catch all while establishing connection

        """

        # TODO: Add ZEO configuration?
        try:
            if os.path.splitext(str(self._config_path))[-1] == ".fs":
                # Handle as local filestorage
                if not self._storage:
                    self._storage = FileStorage.FileStorage(str(self._config_path))
                if not self._db:
                    self._db = DB(self._storage)
            else:
                if not self._db:
                    self._evaluate_config_validity()
                    self._db = config.databaseFromURL(str(self._config_path))
            self._conn = self._db.open()
            self._root = self._conn.root()
            self.logger.append.info(f"DB CONNECTED.")
        except Exception as e:
            self.logger.append.error(e, exc_info=True)
            raise e

    def close_db_connection(self):
        """public method for manually closing connection"""

        self._close()

    def cancel_commit(self) -> None:
        """clear transaction queue"""

        transaction.abort()

    def commit(self) -> None:
        """commit transaction queue"""

        transaction.commit()

    def is_connected(self) -> bool:
        """check if database connection still valid

        Returns
        -------
        bool
            True if still connected

        """

        return (
            isinstance(self.conn, Connection.Connection)
            and not self.conn.getDebugInfo()
        )

    def _evaluate_config_validity(self):
        tree = ET.parse(self._config_path)
        root = tree.getroot()
        if root.tag != "relstorage" or not {child.tag for child in root} & {
            "mysql",
            "oracle",
            "postgres",
            "sqlite3",
        }:
            raise ValueError(
                "Incorrect formatting in zodb config file! Please look at examples in config/templates."
            )

    @property
    def conn(self):
        """returns connection attribute"""

        return self._conn

    @property
    def root(self):
        """returns root, database object"""

        return self._root

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()

    def __del__(self):
        self._close()

    def _close(self):
        if self.is_connected():
            try:
                self._db.pack()
                self._db.close()
                self._conn = None
                self._root = None
            except AttributeError as e:
                self.logger.append.error(e, exc_info=True)
            else:
                self.logger.append.info(f"DB CLOSED.")
