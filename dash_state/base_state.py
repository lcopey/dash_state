"""
>>> from dash_state.utils import try_except

# Simple state value

>>> class State1(BaseState):
...     value: str
>>> State1(value='a value')
State1(value='a value')
>>> class State2(BaseState):
...     value = 'value'
>>> state = State2()
>>> state
State2(value='value')
>>> state.__annotations__
{'value': <class 'str'>}
>>> class SimpleState(BaseState):
...     value: str = ''
>>> SimpleState()
SimpleState(value='')

# Nested state value

>>> class NestedState(BaseState):
...     value: str = ''
...
>>> class State(BaseState):
...     nested = NestedState()
...     value: str
...
>>> state = State(value='a value')
>>> state
State(value='a value', nested=NestedState(value=''))
>>> state.__annotations__
{'value': <class 'str'>, 'nested': <class 'dash_state.base_state.NestedState'>}
>>> state.to_dict()
{'value': 'a value', 'nested': {'value': ''}}
>>> State(nested={'value': 'inner value'}, value='value')
State(value='value', nested=NestedState(value='inner value'))
>>> State.from_dict({'nested': {'value': 'inner_value'}, 'value': 'value'})
State(value='value', nested=NestedState(value='inner_value'))
>>> try_except(
...     lambda : State(nested={'value': 'inner_value'}, value={}),
...     ValueError
... )
'value' expected type <class 'str'> but got <class 'dict'>

# Mutability
>>> class NestedState(BaseState):
...     sequence: list = list()

>>> class State(BaseState):
...     nested: NestedState = NestedState()

>>> state = State()
>>> new_state = State.from_dict(state.to_dict())
>>> state.nested.sequence.append('item')
>>> state.nested.sequence
['item']
>>> new_state.nested.sequence
[]
"""

from dataclasses import dataclass, fields, field as dataclass_field, asdict
from abc import ABCMeta
from typing import Any, Generator
from inspect import isfunction

__all__ = ["BaseState"]


def _is_private_or_special(key: str) -> bool:
    return (key.startswith("__") and key.endswith("__")) or key.startswith("_")


def _is_property_or_method(value: Any):
    return isinstance(value, (property, staticmethod, classmethod)) or isfunction(value)


def is_mutable(value: Any) -> bool:
    """
    Check for mutability of value by checking its hashability.

    >>> is_mutable('aze')
    False
    >>> is_mutable(123)
    False
    >>> is_mutable(list())
    True
    >>> is_mutable(tuple())
    False
    >>> class State(BaseState):
    ...     value: str = ''
    >>> is_mutable(State())
    True
    """
    try:
        hash(value)
        return False
    except TypeError:
        return True


class BaseStateMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        if "__annotations__" not in dct:
            dct["__annotations__"] = dict()

        for key, value in dct.items():
            if not _is_private_or_special(key) and not _is_property_or_method(value):
                # Toutes les valeurs doivent être annotées
                if key not in dct.get("__annotations__", {}):
                    dct["__annotations__"][key] = type(value)

                # Accepte les valeurs mutable par défault en les remplaçant par un champ field au moment
                # de la création de la classe
                if is_mutable(value):
                    dct[key] = dataclass_field(
                        default_factory=lambda: value, kw_only=True
                    )

        new_class = super().__new__(cls, name, bases, dct)
        DataClass = dataclass()
        return DataClass(new_class)


class BaseState(metaclass=BaseStateMeta):
    def items_with_type(self) -> Generator[tuple[str, Any, type], None, None]:
        for field in self.fields:
            yield field, getattr(self, field), self.__annotations__[field]

    def __post_init__(self):
        for field_name, value, Type_ in self.items_with_type():
            if (
                issubclass(Type_, BaseState)
                and not isinstance(value, Type_)
                and isinstance(value, dict)
            ):
                setattr(self, field_name, Type_(**value))
            elif not isinstance(value, Type_):
                raise ValueError(
                    f"{field_name!r} expected type {Type_} but got {type(value)}"
                )

    @property
    def fields(self) -> tuple:
        return tuple(field.name for field in fields(self))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        return cls(**value)
