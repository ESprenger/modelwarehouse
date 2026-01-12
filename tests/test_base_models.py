from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from src.modelwarehouse.structures import DataObject, Model, ModelMeta, Project
from src.modelwarehouse.utils import produce_hash


@pytest.fixture
def base_object() -> DataObject:
    return DataObject()


@pytest.fixture
def base_model_meta() -> ModelMeta:
    return ModelMeta(
        {"model_type": "logistic_regresion", "learning_type": "supervised"}
    )


@pytest.fixture
def yaml_model_meta() -> ModelMeta:
    return ModelMeta("./tests/resources/meta_test.yml")


@pytest.fixture
def base_model_a() -> Model:
    return Model(
        meta_data={},
        model_object=pd.DataFrame(),
        project_name="module_test",
    )


@pytest.fixture
def base_model_b() -> Model:
    return Model(
        meta_data={},
        model_object=pd.DataFrame(np.array([[1, 2], [3, 4]]), columns=["a", "b"]),
        project_name="a_different_test",
    )


@pytest.fixture
def base_project_a() -> Project:
    return Project(
        project_name="module_test", project_description="Empty project for testing."
    )


@pytest.fixture
def base_project_b() -> Project:
    return Project(
        project_name="better_module_test",
        project_description="Empty project for testing.",
    )


class TestBaseObject:
    def test_hash_cmp(self, base_object):
        assert base_object == base_object
        assert base_object.id == base_object.id

    def test_get_field_invalid(self, base_object):
        with pytest.raises(AttributeError) as excinfo:
            base_object.get_field("invalid_attr")
        assert (
            str(excinfo.value)
            == "'DataObject' type does not have attribute - 'invalid_attr' !"
        )


    def test_update_field_invalid(self, base_object):
        with pytest.raises(AttributeError) as excinfo:
            base_object.update_field("invalid_attr", None)
            print(str(excinfo.value))
        assert (
            str(excinfo.value)
            == "'DataObject' type does not have attribute - 'invalid_attr' !"
        )


class TestModelMetaObject:
    def test_construction_via_dict(self, base_model_meta):
        assert base_model_meta.model_type == "logistic_regresion"

    def test_construction_via_path(self, yaml_model_meta):
        assert yaml_model_meta.training_accuracy == 0.97245
        assert (
            str(yaml_model_meta) == "comment=this model was solid., "
            "learning_type=supervised, "
            "model_type=logistic_regression, "
            "objective_func=MSE, "
            "training_accuracy=0.97245"
        )

    def test_get_field_invalid(self, base_model_meta):
        with pytest.raises(AttributeError) as excinfo:
            base_model_meta.get_field("data")
        assert (
            str(excinfo.value) == "'ModelMeta' type does not have attribute - 'data' !"
        )


class TestModelObject:
    def test_model_construction(self):
        existing_meta = ModelMeta({"learning_type":"supervised","model_library":"numpy","model_type":"logistic regression"})
        new_model = Model(project_name="something",meta_data=existing_meta,model_object=np.array([1,2,3,4]))


    def test_hash_cmp(self, base_model_a, base_model_b):

        assert base_model_a.id == produce_hash(
            (
                base_model_a.model_object,
                base_model_a.project_name,
                base_model_a.timestamp,
            )
        )
        assert base_model_a == base_model_a
        assert base_model_a.id == base_model_a.id

        assert base_model_a != base_model_b
        assert base_model_a.id != base_model_b.id

    def test_get_field_invalid(self, base_model_a):
        with pytest.raises(AttributeError):
            base_model_a.get_field("data")

    def test_update_field_invalid(self, base_model_a):
        with pytest.raises(AttributeError) as excinfo:
            base_model_a.update_field("timestamp", datetime.now())
        assert str(excinfo.value) == "'timestamp' is an immutable field !"


class TestProjectObject:
    def test_hash_cmp(self, base_project_a, base_project_b):
        assert base_project_a.id == produce_hash((base_project_a.project_name,))
        assert base_project_a == base_project_a

        assert base_project_a != base_project_b
        assert base_project_a.id != base_project_b.id

    def test_add_remove_model(self, base_project_a, base_model_a, base_model_b):
        base_project_a.add_model(base_model_a.id)
        assert list(base_project_a.model_ids) == [base_model_a.id]

        with pytest.raises(AttributeError) as excinfo:
            base_project_a.add_model(base_model_a.id)
        assert (
            str(excinfo.value)
            == f"Model - '{base_model_a.id}' already exists in project - '{base_project_a.id}' - '{base_project_a.project_name}'!"
        )

        with pytest.raises(AttributeError) as excinfo:
            base_project_a.remove_model(base_model_b.id)
            assert (
                str(excinfo.value)
                == f"Model - '{base_model_b.id}' is not in project - '{base_project_a.id}' - '{base_project_a.project_name}'!"
            )

        base_project_a.add_model(base_model_b.id)
        assert list(base_project_a.model_ids) == [base_model_a.id, base_model_b.id]

    def test_update_field_invalid(self, base_model_a):
        with pytest.raises(AttributeError):
            base_model_a.update_field("project_name", "better_module_test")
