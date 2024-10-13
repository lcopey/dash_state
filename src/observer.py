"""
>>> from src.state import BaseState
>>> class State(BaseState):
...     class Input(BaseState):
...         value: str
...     input: Input
...     value: str
>>> Observer(State)
<Observer <class 'src.observer.State'>>
>>> Observer(State).on.input
<Proxy <class 'src.observer.State.Input'>>
>>> try:
...     Observer(State).on.wrong_input
... except ValueError as e:
...     e
ValueError("wrong_input does not exist in the field of <class 'src.observer.State'>")
>>> Observer(State).on.input.value
<Proxy <class 'src.observer.State.Input'>>
>>> try:
...     Observer(State).on.input.wrong_input
... except ValueError as e:
...     e
ValueError("wrong_input does not exist in the field of <class 'src.observer.State.Input'>")
>>> Observer(State).on.input._path
['input']
>>> Observer(State).on.input.value._path
['input', 'value']
"""

from typing import TypeVar, Generic

from .state import BaseState

T = TypeVar("T", bound=BaseState)


class Proxy(Generic[T]):
    def __init__(self, cls: type[T], path: list = None):
        self._path = path or list()
        self._cls = cls
        self._valid_values = set(cls.__annotations__.keys())

    def __getattribute__(self, item: str):
        if item.startswith("_"):
            return super().__getattribute__(item)

        if item in self._valid_values:
            self._path.append(item)
            obj = self._cls.__annotations__.get(item)
            if issubclass(obj, BaseState):
                value = Proxy(obj, self._path)
            else:
                value = self

        else:
            raise ValueError(f"{item} does not exist in the field of {self._cls}")
        return value

    def __repr__(self):
        return f"<Proxy {self._cls!r}>"


class Observer(Generic[T]):
    def __init__(self, cls: type[T]):
        self._cls = cls

    # Retourne T pour tromper les annotations de type et avoir un peu d'autocompletion
    # dans l'éditeur
    @property
    def on(self) -> T:
        return Proxy(self._cls)  # type: ignore

    def __repr__(self):
        return f"<Observer {self._cls!r}>"

    @property
    def tree(self):
        """
        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         value: str
        ...     input: Input
        ...     value: str
        >>> Observer(State).tree
        [{'input': ['value']}, 'value']
        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         class InnerState(BaseState):
        ...             value: str
        ...         value: str
        ...         other_value: InnerState
        ...     input: Input
        ...     value: str
        >>> Observer(State).tree
        [{'input': ['value', {'other_value': ['value']}]}, 'value']

        Returns:

        """
        current_tree = []
        for field in self._cls.fields:
            obj = self._cls.__annotations__[field]
            if issubclass(obj, BaseState):
                current_tree.append({field: Observer(obj).tree})
            else:
                current_tree.append(field)
        return current_tree

    def all_paths(
        self, path: tuple | None = None, indexes: list | None = None
    ) -> list[tuple]:
        """
        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         value: str
        ...     input: Input
        ...     value: str
        >>> Observer(State).all_paths()
        [('input',), ('input', 'value'), ('value',)]
        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         class InnerState(BaseState):
        ...             value: str
        ...         value: str
        ...         other_value: InnerState
        ...     input: Input
        ...     value: str
        >>> Observer(State).all_paths()
        [('input',), ('input', 'value'), ('input', 'other_value'), ('input', 'other_value', 'value'), ('value',)]

        Returns:

        """
        indexes = indexes or list()
        path = path or tuple()
        for field in self._cls.fields:
            obj = self._cls.__annotations__.get(field)
            value = (*path, field)
            if issubclass(obj, BaseState):
                indexes.append(value)
                Observer(obj).all_paths(path=value, indexes=indexes)
            else:
                indexes.append(value)
        return indexes
