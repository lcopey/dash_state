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
