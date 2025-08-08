from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from BTrees.IOBTree import IOBTree
from modelwarehouse.controller import Depot
from modelwarehouse.structures.core import Model, Project
from modelwarehouse.utils.core import produce_hash

from .util import clear_filestorage, gen_test_path, TEST_PATH


@pytest.fixture(scope="module")
def pre_depot() -> Depot:
    clear_filestorage()
    return Depot(
        path_to_configuration=gen_test_path("test_db.fs"),
        log_filename="test.log",
        log_filepath=TEST_PATH,
    )


@pytest.fixture(scope="module")
def model_a() -> Model:
    return Model(
        model_object=pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"]),
        project_name="test_project",
        meta_data=gen_test_path("meta_test.yml"),
    )


@pytest.fixture(scope="module")
def model_b() -> Model:
    return Model(
        model_object=np.array([1, 2, 3, 4]),
        project_name="test_project",
        meta_data={"model_type": "random_forest", "learning_type": "reinforcement"},
    )


@pytest.fixture(scope="module")
def model_c() -> Model:
    return Model(
        model_object=np.array([1, 2, 3, 4]),
        project_name="second_project",
        meta_data={"model_type": "logistic_regression", "learning_type": "supervised"},
    )


@pytest.fixture(scope="module")
def project_a() -> Project:
    return Project(project_name="test_project")


@pytest.fixture(scope="module")
def project_b() -> Project:
    return Project(project_name="second_project")


@pytest.fixture(scope="module")
def post_depot() -> Depot:
    clear_filestorage("test_db_v2")
    dep = Depot(
        path_to_configuration=gen_test_path("test_db_v2.fs"),
        log_filename="test.log",
        log_filepath=TEST_PATH,
    )
    return dep


class TestHandlerDepotSimple:
    def test_root_object_init(self, pre_depot):
        assert isinstance(pre_depot.models, IOBTree)
        assert isinstance(pre_depot.projects, IOBTree)

    def test_add_model_no_project(self, pre_depot, model_a):
        pre_depot.add_model(model_a)
        assert model_a.id not in pre_depot.models

    def test_add_project(self, pre_depot, project_a):
        pre_depot.add_project(project_a)
        assert project_a.id in pre_depot.projects

    def test_add_model_w_project(self, pre_depot, model_a):
        pre_depot.add_model(model_a)
        model_a_project_id = produce_hash(model_a.project_name)
        assert model_a.id in pre_depot.models
        assert model_a.id in pre_depot.projects[model_a_project_id].models

    def test_add_additional_model_w_project(self, pre_depot, model_a, model_b):
        pre_depot.add_model(model_b)
        model_b_project_id = produce_hash(model_b.project_name)
        assert model_b.id in pre_depot.models
        assert model_b.id in pre_depot.projects[model_b_project_id].models
        assert sorted(pre_depot.projects[model_b_project_id].model_ids) == sorted(
            [m.id for m in [model_a, model_b]]
        )

    def test_remove_existing_model(self, pre_depot, model_b):
        pre_depot.remove_model(model_b.id)
        model_b_project_id = produce_hash(model_b.project_name)
        assert model_b.id not in pre_depot.models
        assert model_b.id not in pre_depot.projects[model_b_project_id].models

    def test_remove_nonexistent_model(self, pre_depot, model_b):
        pre_depot.remove_model(model_b.id)
        assert model_b.id not in pre_depot.models

    def test_update_attr(self, pre_depot, model_a):
        pre_depot.update_object_attr(
            model_a.id,
            "model_object",
            pd.DataFrame([[5, 6], [7, 8]], columns=["a", "b"]),
        )
        assert (
            (pre_depot.models[model_a.id])
            .get_field("model_object")
            .equals(pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"]))
        )

    def test_update_nested_attr(self, pre_depot, model_a):
        pre_depot.update_object_attr(model_a.id, "learning_type", "unsupervised")
        assert (
            pre_depot.models[model_a.id]
            .get_field("meta_data")
            .get_field("learning_type")
            == "unsupervised"
        )

    def test_move_model_new_project(self, pre_depot, project_b, model_c):
        second_project_id = produce_hash(model_c.project_name)
        pre_depot.add_project(project_b)
        pre_depot.add_model(model_c)
        pre_depot.move_model_to_project(model_c.id, "test_project")
        assert pre_depot.projects[second_project_id].get_field("models") == []
        assert model_c.id not in pre_depot.models

    def test_remove_existing_project(self, pre_depot, project_a):
        project_a_id = project_a.id
        model_ids = pre_depot.projects[project_a_id].get_field("models")
        pre_depot.remove_project(project_a_id)
        assert project_a_id not in pre_depot.projects
        for model_id in model_ids:
            assert model_id not in pre_depot.models

    def test_remove_nonexistent_project(self, pre_depot, project_a):
        pre_depot.remove_project(project_a.id)
        assert project_a.id not in pre_depot.projects
