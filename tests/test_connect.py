from __future__ import annotations

from types import NoneType

import pytest
from modelwarehouse.database import ConnectionManager
from persistent.mapping import PersistentMapping
from ZConfig import ConfigurationSyntaxError
from ZODB.Connection import Connection
from ZODB.DB import DB

from .util import TEST_PATH, gen_test_path


@pytest.fixture
def base_manager_fs() -> ConnectionManager:
    return ConnectionManager(gen_test_path("test_db.fs"), "test.log", TEST_PATH, False)


class TestConnectionManager:
    def test_not_connected(self, base_manager_fs):
        assert not base_manager_fs.is_connected()

    def test_is_connected(self, base_manager_fs):
        base_manager_fs.create_db_connection()
        assert base_manager_fs.is_connected()

    def test_create_db_connection(self, base_manager_fs):
        base_manager_fs.create_db_connection()
        assert isinstance(base_manager_fs.conn, Connection)
        assert isinstance(base_manager_fs.root, PersistentMapping)

    def test_close_db_connection(self, base_manager_fs):
        base_manager_fs.create_db_connection()
        base_manager_fs.close_db_connection()
        assert isinstance(base_manager_fs.conn, NoneType)
        assert isinstance(base_manager_fs.root, NoneType)
        assert isinstance(base_manager_fs._db, DB)

    def test_context_manager(self, base_manager_fs):
        base_manager_fs.create_db_connection()
        with base_manager_fs as _:
            _ = 1
        assert not base_manager_fs.is_connected()

    @pytest.mark.parametrize(
        "config_path,expectation",
        [
            (gen_test_path("bad_config_a.tmpl"), pytest.raises(ValueError)),
            (gen_test_path("bad_config_b.tmpl"), pytest.raises(ValueError)),
            (
                gen_test_path("valid_config.tmpl"),
                pytest.raises(ConfigurationSyntaxError),  # Pass assesrtions
            ),
        ],
    )
    def test_check_config_validity(self, config_path, expectation):
        conn = ConnectionManager(config_path, "test.log", TEST_PATH, False)
        with expectation:
            assert conn.create_db_connection() is not None
