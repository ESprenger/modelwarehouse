from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from BTrees.IOBTree import IOBTree
from modelwarehouse.database import ConnectionManager
from modelwarehouse.structures import Model, Project
from modelwarehouse.utils import MWLogger, produce_hash, resolve_search


def safe_transaction(supress_abort: bool = False):
    """Wrap mutating function in try/except that handles logging + database transaction management.

    Parameters
    ----------
    supress_abort : boolean
        Option to skip abort transactions upon failure.

    """

    def safe_transaction_decorator(fn):
        @functools.wraps(fn)
        def safe_transaction_wrapper(depot, *args, **kwargs):
            try:
                fn(depot, *args, **kwargs)
                depot.conn_manager.commit()
                depot.logger.append.info(
                    f"Successful COMMIT - {fn} - {args} - {['{}={}'.format(*items) for items in kwargs.items()]}"
                )
            except Exception as err:
                if not supress_abort:
                    depot.conn_manager.cancel_commit()
                    depot.logger.append.error(err, stack_info=True, exc_info=True)

        return safe_transaction_wrapper

    return safe_transaction_decorator


class Depot:
    """
    A class for handling database connectivity + read/write functionality.

    ...

    Attributes
    ----------
    logger : MWLogger
         logging setup
    conn_manager : ConnectionManager
        database connection manager

    Notes
    -----
    This is the main API for interfacing/operability with ZODB database.

    """

    def __init__(
        self,
        path_to_configuration: str | Path,
        log_filename: str = "modelwarehouse.depot",
        log_filepath: Optional[str | Path] = None,
    ) -> None:
        """Init method.

        Parameters
        ----------
        path_to_configuration : str | Path
            path (str | Path) to database config or filestorage
        log_filename : str
            name of log file
        log_filepath : Optional[str | Path]
            optional dir (str | Path) for log file

        Examples
        --------
        >>> my_depot = Depot("/path/to/db/config","name_of_log_file","/path/to/log")

        """

        self.logger = MWLogger(
            filename=log_filename, filepath=log_filepath, level="info"
        )
        self.conn_manager = ConnectionManager(
            path_to_configuration=path_to_configuration,
            log_filename=log_filename,
            log_filepath=log_filepath,
        )
        self._validate_root_objects()

    def _validate_root_objects(self) -> None:
        for k in ["models", "projects"]:
            self._init_root_object(k)

    @safe_transaction(supress_abort=True)
    def _init_root_object(self, key: str) -> None:
        if key in self.conn_manager.root:
            raise KeyError(f"Tree '{key}' already exists !")
        self.conn_manager.root[key] = IOBTree()

    def reset_connection(self) -> None:
        """Reset database connection through 'conn_manager' field"""

        if not self.conn_manager.is_connected():
            self.conn_manager.create_db_connection()
            self._validate_root_objects()

    @safe_transaction()
    def add_model(self, new_model: Model) -> None:
        """Add new model to database

        Parameters
        ----------
        new_model : Model
            new model object to write

        Raises
        ------
        KeyError
            associated project of 'new_model' doesn't exist in database

        KeyError
            'new_model' already exists in database

        Examples
        --------
        >>> my_depot = Depot(...)
        >>> my_model = Model(...)
        >>> my_depot.add_model(my_model)

        """

        model_id = new_model.id
        project_name = new_model.project_name
        project_id = produce_hash(project_name)

        if project_id not in self.projects:
            raise KeyError(
                f"Project - '{project_name}' - does not exist!  Add project to database first!"
            )

        if model_id in self.models:
            raise KeyError(
                f"Model - '{model_id}' - already exists! Model = {new_model}"
            )

        self.models[model_id] = new_model
        self.projects[project_id].add_model(model_id)
        self.logger.append.info(
            f"Add model - '{model_id}' to project - '{project_name}'"
        )

    @safe_transaction()
    def remove_model(self, model_id: int):
        """Remove model from database by 'Model' id.

        Parameters
        ----------
        model_id : int
            integer key for 'Model' object

        Raises
        ------
        KeyError
            'model_id' key is not in database

        Examples
        --------
        >>> my_depot = Depot(...)
        >>> my_depot.remove_model(model_id=154)

        """

        if model_id not in self.models:
            raise KeyError(f"Model - '{model_id}' - does not exist!")

        project_id = produce_hash(self.models[model_id].get_field("project_name"))

        del self.models[model_id]
        (self.projects[project_id]).remove_model(model_id)
        self.logger.append.info(
            f"Remove model - '{model_id}' from project - '{self.projects[project_id].project_name}'"
        )

    @safe_transaction()
    def add_project(self, new_project: Project):
        """Add new project to database.

        Parameters
        ----------
        new_project : Project
            instantiated 'Project' object

        Raises
        ------
        KeyError
            project already exists in database

        Examples
        --------
        >>> my_depot = Depot(...)
        >>> new_project = Project(...)
        >>> my_depot.add_project(new_project)

        """

        if new_project.id in self.projects:
            raise KeyError(f"Project - {new_project.id} - already exists!")

        self.projects[new_project.id] = new_project
        self.logger.append.info(
            f"Add project - '{new_project.project_name}' - {new_project.id}"
        )

    @safe_transaction()
    def remove_project(
        self,
        project_id: int,
        move_to_new_project: Optional[str | int] = None,
    ):
        """Remove project and associated models from database

        Parameters
        ----------
        project_id : int
            int ID of project
        move_to_new_project : Optional[str]
            new project (str | int) to move models within 'project_id' to

        Raises
        ------
        KeyError
            'project_id' is not in database
        """

        if project_id not in self.projects:
            raise KeyError(f"Project - {project_id} - does not exist!")

        for model_id in self.projects[project_id].get_field("models"):
            if move_to_new_project:
                self.move_model_to_project(model_id, move_to_new_project)
            else:
                self.remove_model(model_id)

        del self.projects[project_id]
        self.logger.append.info(f"Remove project - {project_id}")

    @safe_transaction()
    def update_object_attr(self, id: int, attr: str, val: Any) -> None:
        """Update mutable field of database object

        Parameters
        ----------
        id : int
            either project or model integer ID
        attr : str
            mutable field of object (Model | Project)
        val : Any
            new value to set

        Raises
        ------
        AttributeError
            'id' does not exist as key in either 'projects' or 'models' trees

        Notes
        -----
        if 'attr' exists but is an immutable field, KeyError will be raised
        by object being mutated
        """

        if id in self.models:
            self.models[id].update_field(attr, val)
            self.logger.append.info(f"For model - {id} - updating '{attr}' with '{val}")
        elif id in self.projects:
            self.projects[id].update_field(attr, val)
            self.logger.append.info(
                f"For project - {id} - updating '{attr}' with '{val}"
            )
        else:
            raise AttributeError(
                f"Object field - {id} - does not exist in 'projects' or 'models' !"
            )

    def move_model_to_project(self, model_id: int, new_project: str | int):
        """Move model from project to a different project

        Parameters
        ----------
        model_id : int
            integer ID of model to be moved
        new_project : str | int
            destination project, either project name or id

        Examples
        --------
        >>> my_depot = Depot(...)
        >>> model_2_move = 154

        >>> new_project_id = 9765
        >>> my_depot.move_model_to_project(model_id=model_2_move, new_project=new_project_id)

        >>> new_project_name = "a_new_project"
        >>> my_depot.move_model_to_project(model_id=model_2_move, new_project=new_project_name)

        """
        existing_model = self.models[model_id]
        new_project = (
            new_project
            if isinstance(new_project, str)
            else self.projects[new_project].get_field("project_name")
        )

        new_model_obj = Model(
            project_name=new_project,
            model_object=existing_model.get_field("model_object"),
            meta_data=existing_model.get_field("meta_data"),
        )
        self.add_model(new_model_obj)
        self.remove_model(model_id)

    def search_models(
        self, view_only: bool = True, project: Optional[str | int] = None, **kwargs
    ) -> List[Tuple[int, str | Model]]:
        """Search database for models

        Parameters
        ----------
        view_only : bool
            return str representations instead of raw objects
        project : str | int, optional
            limit search to within single project
        **kwargs:
            abritray keyword search parameters

        Returns
        -------
        List[Tuple[int, str | Model]]
            list of tuple pairs (str, str | Model)

        Examples
        --------
        >>> my_depot = Depot(...)

        >>> my_depot.search(project=12345,learning_type='==supervised',timestamp="<2020-01-01 12:00:00",test_accuracy=">=0.95")

        """

        format_output = lambda m: str(m) if view_only else m
        if project:
            project_id = project if isinstance(project, int) else produce_hash(project)
            iter_object = (
                (model.id, model)
                for model in self.projects[project_id].get_field("models")
            )
        else:
            iter_object = self._traverse(
                root_tree="models", include_values=True, lazy=True
            )
        return [
            (id, format_output(model))
            for id, model in iter_object
            if self._inspect_model(model, **kwargs)
        ]

    def _inspect_model(self, model_object: Model, **kwargs) -> bool:
        for key, val in kwargs.items():
            comp_op, formatted_val = resolve_search(val)
            try:
                if not comp_op(model_object.get_field(key), formatted_val):
                    return False
            except AttributeError as _:
                return False
        return True

    def _traverse(
        self, root_tree: str, include_values: bool = False, lazy: bool = True
    ) -> Iterable:
        tree = self.projects if root_tree == "projects" else self.models

        if lazy:
            return tree.iteritems() if include_values else tree.iterkeys()

        return tree.items() if include_values else tree.keys()

    @property
    def projects(self) -> IOBTree:
        """Return 'projects' BTree from database

        Returns
        -------
        IOBTree

        """
        return self.conn_manager.root["projects"]

    @property
    def models(self) -> IOBTree:
        """Return 'models' BTree from database

        Returns
        -------
        IOBTree

        """
        return self.conn_manager.root["models"]

    @property
    def project_names(self) -> List[str]:
        """Return sorted list of project names

        Returns
        -------
        List[str]
            list of str

        """

        return sorted(
            [
                v.get_field("project_name")
                for _, v in self._traverse(root_tree="projects", include_values=True)
            ]
        )
