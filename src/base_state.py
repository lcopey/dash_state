"""
>>> def try_except(func, error):
...     try:
...         func()
...     except error as e:
...         print(e)

# Simple state value
>>> class State(BaseState):
...     value: str
>>> State(value='a value')
State(value='a value')
>>> class State(BaseState):
...     value = 'value'
>>> state = State()
>>> state
State(value='value')
>>> state.__annotations__
{'value': <class 'str'>}

# Nested state value
>>> class Input(BaseState):
...     value: str = ''
...
>>> class State(BaseState):
...     input_: Input = Input()
...     value: str
...
>>> state = State(value='a value')
>>> state
State(input_=Input(value=''), value='a value')
>>> state.__annotations__
{'input_': <class 'src.base_state.Input'>, 'value': <class 'str'>}
>>> state.to_dict()
{'input_': {'value': ''}, 'value': 'a value'}
>>> State(input_={'value': 'inner value'}, value='value')
State(input_=Input(value='inner value'), value='value')
>>> State.from_dict({'input_': {'value': 'inner_value'}, 'value': 'value'})
State(input_=Input(value='inner_value'), value='value')
>>> try_except(
...     lambda : State(input_={'value': 'inner_value'}, value={}),
...     ValueError
... )
'value' expected type <class 'str'> but got <class 'dict'>
"""

from dataclasses import dataclass, fields, field as dataclass_field, asdict
from abc import ABCMeta
from typing import Any, Generator
from inspect import isfunction


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
        return dataclass()(new_class)


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
