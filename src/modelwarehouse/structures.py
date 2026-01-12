from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, ClassVar, Iterable, List, Optional, Tuple, Union

import pandas as pd
import persistent
import yaml
from modelwarehouse.utils import infer_obj_module, produce_hash


class DataObject(persistent.Persistent):
    """
    A base class for persistent+dataclass objects.

    """

    @property
    def id(self):
        return self._eval_hash()

    def get_field(self, key: str) -> Any:
        val = self[key]
        if isinstance(val,list):
            return val.copy()
        return val

    def update_field(self, key: str, val: Any) -> None:
        self[key] = val

    def __eq__(self, other) -> bool:
        return isinstance(self, self.__class__) and self.id == other.id

    def __lt__(self, other) -> bool:
        return isinstance(self, self.__class__) and self.id < other.id

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __getitem__(self, key: str) -> Any:
        try:
            valid_object = self._filter_data_object(key)
        except StopIteration:
            raise AttributeError(
                f"'{self.__class__.__name__}' type does not have attribute - '{key}' !"
            )

        return getattr(valid_object, key)

    def __setitem__(self, key: str, val: Any) -> None:
        try:
            valid_object = self._filter_data_object(key)
        except StopIteration:
            raise AttributeError(
                f"'{self.__class__.__name__}' type does not have attribute - '{key}' !"
            )
        else:
            if (
                "_static_fields_" in valid_object
                and key in valid_object._static_fields_
            ):
                raise AttributeError(f"'{key}' is an immutable field !")
        setattr(valid_object, key, val)

    def _find_data_objects(self) -> Iterable:
        return (obj for obj in self.__dict__.values() if isinstance(obj, DataObject))

    def _filter_data_object(self, key: str) -> DataObject:
        return next(
            obj for obj in chain(self._find_data_objects(), (self,)) if key in obj
        )

    def _eval_hash(self) -> int:
        if "_static_fields_" in self:
            return produce_hash(
                tuple(self.get_field(field) for field in sorted(self._static_fields_))
            )

        # Deterministic int ID backup within given python session
        return super().__hash__()


@dataclass(init=False, eq=False, repr=False)
class ModelMeta(DataObject):
    """
    A class for storing Meta information about 'Model' object.

    ...

    Attributes
    ----------
    model_type : str, optional(default=None)
         model type of model object
    learning_type : str, optional(default=None)
        type of ML learning (supervised, unsupervised etc)
    model_library : str, optional(default=None)
        inferred library from Model.model_object
    dataset : str, optional(default=None)
        information for dataset used in training
    objective_func : str, optional(default=None)
        objective function used in ML training
    training_accuracy : float, optional(default=None)
        quantiative value for training accuracy
    test_accuracy : logging.FileHandlerf
        file oriented logging handler
    comment : logging.FileHandler
        file oriented logging handler

    Notes
    -----
    Object can include any or no fields.  The 'model_library' field will be set by the
    'Model' object that contains the 'ModelMeta' object.

    """

    model_type: Optional[str] = field(init=False, default=None)
    learning_type: Optional[str] = field(init=False, default=None)
    model_library: Optional[str] = field(init=False, default=None)
    dataset: Optional[str] = field(init=False, default=None)
    objective_func: Optional[str] = field(init=False, default=None)
    training_accuracy: Optional[float] = field(init=False, default=None)
    test_accuracy: Optional[float] = field(init=False, default=None)
    comment: Optional[str] = field(init=False, default=None)

    def __init__(self, data: str | dict):
        """Init method.  Dataclass auto init is disabled.

        Parameters
        ----------
        data : str | dict
            either a dict or path to dict like object (yaml)

        """
        self._parse_input(data)

    def _load_input(self, input: str | dict) -> dict:
        if isinstance(input, dict):
            return input
        with open(Path(input), "r") as stream:
            return yaml.safe_load(stream)

    def _parse_input(self, input: str | dict) -> None:
        for key, value in self._load_input(input).items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return ", ".join(
            [
                f"{field}={self.__dict__[field]}"
                for field in sorted(self.__dict__.keys())
            ]
        )


@dataclass(eq=False, repr=False)
class Model(DataObject):
    """
    A class for storing Meta information about 'Model' object.

    ...

    Attributes
    ----------
    model_object : Any
         trained model object (pytorch, sklearn etc)
    project_name : str
        associated project model belongs to
    meta_data : ModelMeta | str | dict
        either ModelMeta object or dict or path to dict like object (yaml)
    creator : str
        inferred information from calling OS user
    timestamp : Timestamp
        pandas timestamp, evaluated when object initialized

    Notes
    -----
    Fields of 'model_object', 'project_name', and 'meta_data' are required.

    Static_fields denote immutable fields.  Unique ID is dependent on these.

    """

    _static_fields_: ClassVar[Tuple] = ("timestamp", "project_name", "model_object")

    model_object: Any
    project_name: str
    meta_data: Union[ModelMeta, str, dict]
    creator: str = field(init=False, default="Unknown")
    timestamp: pd.Timestamp = field(
        default_factory=lambda: pd.to_datetime(datetime.now())
    )

    def __post_init__(self):
        if isinstance(self.meta_data,(str, dict)):
            self.update_field("meta_data", ModelMeta(self.meta_data))

        self.meta_data.update_field(
            "model_library", infer_obj_module(self.model_object)
        )
        self.update_field("creator", os.getlogin())

    def __repr__(self) -> str:
        return ", ".join(
            [
                f"model_id={self.id}",
                f"project_name={self.project_name}",
                f"creator={self.creator}",
                f"timestamp={self.timestamp}",
                f"meta_data=[{self.meta_data}]",
            ]
        )


@dataclass(eq=False, repr=False)
class Project(DataObject):
    """
    A class for storing information about ML Project.

    ...

    Attributes
    ----------
    project_name : str
         name of project
    project_description : str, optional(default=None)
        optional commentary about project
    models : List(int)
        list of 'Model' object IDs in object database
    creator : str
        inferred information from calling OS user

    Notes
    -----
    Field of 'project_name' is required.

    Static_fields denote immutable fields.  Unique ID is dependent on these.

    """

    _static_fields_: ClassVar[Tuple] = ("project_name",)

    project_name: str
    project_description: Optional[str] = None
    models: List[int] = field(default_factory=list)
    creator: str = field(init=False, default="Unknown")

    def __post_init__(self):
        self.update_field("creator", os.getlogin())

    @property
    def model_ids(self):
        """list(int) : Return list of Model integer IDs."""
        return self.models

    def add_model(self, new_model_id: int):
        """
        Add model ID to 'models' field.

        Parameters
        ----------
        new_model_id : int
            int ID of new model

        Raises
        ------
        AttributeError
            'new_model_id' already exists in project's 'models' field.

        Examples
        --------
        >>> id_2_add = 154
        >>> project_obj.add_model(154)

        Notes
        -----
        This method should be called from a handling object that has an established
        database connection in scope.

        """

        if new_model_id in self.models:
            raise AttributeError(
                f"Model - '{new_model_id}' already exists in project - '{self.id}' - '{self.project_name}'!"
            )
        self.models.append(new_model_id)
        self._p_changed = True

    def remove_model(self, model_id: int):
        """
        Remove model ID from 'models' field.

        Parameters
        ----------
        model_id : int
            Int ID to be removed.

        Raises
        ------
        AttributeError
            If 'model_id' does not exist in 'models' field.

        Examples
        --------
        >>> id_2_remove = 154
        >>> project_obj.remove_model(154)

        Notes
        -----
        This method should be called from a handling object that has an established
        database connection in scope.

        """

        if model_id not in self.models:
            raise AttributeError(
                f"Model - '{model_id}' is not in project - '{self.id}' - '{self.project_name}'!"
            )
        self.models.remove(model_id)
        self._p_changed = True

    def __repr__(self) -> str:
        return ", ".join(
            [
                f"project_name={self.project_name}",
                f"project_description={self.project_description}",
                f"creator={self.creator}",
                f"models={self.models}",
            ]
        )
