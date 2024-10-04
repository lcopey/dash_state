from functools import reduce
from abc import ABCMeta
from typing import Any
from inspect import Signature


def _is_private_or_special(key: str) -> bool:
    return (key.startswith("__") and key.endswith("__")) or key.startswith("_")


def _get_default_kwargs(dct: dict[str, Any]):
    return {
        key: value
        for key, value in dct.items()
        if not _is_private_or_special(key) and not isinstance(value, BaseStateMeta)
    }


def _get_annotations(dct: dict[str, Any], defaults_kwargs: dict[str, Any]):
    annotations = {
        key: value
        for key, value in dct.get("__annotations__", dict()).items()
        if not _is_private_or_special(key)
    }
    for key, value in defaults_kwargs.items():
        if key not in annotations:
            annotations[key] = type(value)

    return annotations


def _check_required_arg(annotations, default_kwargs: dict, kwargs: dict):
    error_msg = []
    for required in annotations.keys():
        if required not in kwargs and required not in default_kwargs:
            error_msg.append(f"{required}: field required")
    if error_msg:
        raise ValueError("\n".join(error_msg))


def _get_fields(annotations: dict, default_kwargs: dict):
    fields = [field for field in annotations.keys()]
    fields.extend([field for field in default_kwargs.keys() if field not in fields])
    return fields


class BaseStateMeta(ABCMeta):
    def __new__(cls, name, bases, dct):
        """

        >>> class BaseState(metaclass=BaseStateMeta):
        ...     input: str
        ...     value: float = 10
        ...     object = set()
        >>> try:
        ...     BaseState()
        ... except ValueError as e:
        ...     e
        ValueError('input: field required')
        >>> BaseState(input='str')._fields
        ['input', 'value', 'object']
        >>> BaseState(input='str').__annotations__
        {'input': <class 'str'>, 'value': <class 'float'>, 'object': <class 'set'>}

        Args:
            name:
            bases:
            dct:
        """
        default_kwargs = _get_default_kwargs(dct)
        annotations = _get_annotations(dct, default_kwargs)
        dct["_default"] = default_kwargs
        dct["_fields"] = _get_fields(annotations, default_kwargs)
        dct["__annotations__"] = annotations
        self_instance = super().__new__(cls, name, bases, dct)

        def _from_kwargs(self, kwargs: dict):
            for key, value in kwargs.items():
                if key in annotations or key in default_kwargs:
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unexpected keyword {key}")

        def _set_default(self, default_kwargs: dict):
            for key, value in default_kwargs.items():
                attr = getattr(self, key, None)
                if not attr:
                    setattr(self, key, value)

        def new_init(self, *args: dict, **kwargs):
            _check_required_arg(annotations, default_kwargs, kwargs)
            # for base in bases:
            #     for key, obj in base.__dict__.items():
            #         if not _is_private_or_special(key):
            #             setattr(self, key, obj)
            if not kwargs and (args and len(args) == 1):
                raise NotImplemented()
            elif kwargs and not args:
                _from_kwargs(self, kwargs)
                _set_default(self, default_kwargs)
            elif not kwargs and not args:
                _set_default(self, default_kwargs)
            else:
                raise TypeError(
                    "Either one dict argument or multiple keyword arguments only"
                )

        self_instance.__init__ = new_init
        return self_instance


class BaseState(metaclass=BaseStateMeta):
    """
    >>> class State(BaseState):
    ...     input: str
    ...     value: float = 10
    ...     object = set()
    >>> try:
    ...     State()
    ... except ValueError as e:
    ...     e
    ValueError('input: field required')

    >>> State(input='str')
    State(input='str', value=10, object=set())

    >>> try:
    ...     State(input='str', wrong_keyword=10)
    ... except ValueError as e:
    ...     e
    ValueError('Unexpected keyword wrong_keyword')

    >>> class State(BaseState):
    ...     class Input(BaseState):
    ...         value: str
    ...
    ...     input: Input
    >>> state = State(input=State.Input(value='str'))
    >>> state
    State(input=Input(value='str'))
    >>> state.input
    Input(value='str')
    >>> state.input.value
    'str'

    """

    _fields: list
    _default: dict[str, Any]

    def __repr__(self):
        attrs = ", ".join(f"{field}={getattr(self, field)!r}" for field in self._fields)
        return f"{self.__class__.__name__}({attrs})"

    @classmethod
    def default(cls):
        """

        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         value: str
        ...
        ...     input: Input
        ...     value: str
        ...     object = set()

        >>> State.default()
        State(input=Input(value=None), value=None, object=set())

        Returns:

        """
        result = {}
        for key, value in cls._default.items():
            result[key] = value
        for key, Type in cls.__annotations__.items():
            if key not in result:
                if issubclass(Type, BaseState):
                    result[key] = Type.default()
                else:
                    result[key] = None

        return cls(**result)

    @classmethod
    def from_dict(cls, values: dict):
        """

        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         value: str
        ...
        ...     input: Input
        ...     value: str
        >>> try:
        ...     State.from_dict({'wrong_input': 'value'})
        ... except ValueError as e:
        ...     e
        ValueError('Unknown field wrong_input')

        >>> State.from_dict({'input': {'value': 'str'}, 'value': 'aze'})
        State(input=Input(value='str'), value='aze')

        Args:
            values:

        Returns:

        """
        result = {}
        for key, value in values.items():
            Type = cls.__annotations__.get(key)
            if not Type:
                raise ValueError(f"Unknown field {key}")
            if issubclass(Type, BaseState):
                result[key] = Type.from_dict(value)
            else:
                result[key] = value

        return cls(**result)

    def as_dict(self):
        """

        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         value: str
        ...
        ...     input: Input
        >>> State(input=State.Input(value='str')).as_dict()
        {'input': {'value': 'str'}}

        """
        result = {}
        for field in self._fields:
            attr = getattr(self, field)
            if isinstance(attr, BaseState):
                result[field] = attr.as_dict()
            else:
                result[field] = attr
        return result


def merge(
    left: dict, right: dict, raise_on_left_missing: bool = True, path: tuple = ()
):
    """
    >>> left = {'input': {'value': 'test'}}
    >>> right = {'input': {'value': 'test2'}}
    >>> merge(left, right)
    {'input': {'value': 'test2'}}

    >>> left = {'input': {'inner': {'value': 'test'}}}
    >>> right = {'input': {'inner': {'value': 'test2'}}}
    >>> merge(left, right)
    {'input': {'inner': {'value': 'test2'}}}

    >>> left = {'input': {'value': 'test'}}
    >>> right = {'input': {'value': 'test2', 'value2': 'test'}}
    >>> try:
    ...     merge(left, right)
    ... except ValueError as e:
    ...     e
    ValueError("La clé input.value2 n'existe pas dans le dictionnaire de gauche")
    >>> left = {'input': {'value': 'test'}}
    >>> right = {'input': {'value': 'test2', 'value2': 'test'}}
    >>> merge(left, right, raise_on_left_missing=False)
    {'input': {'value': 'test2', 'value2': 'test'}}

    >>> left = {'input': {'value': 'test', 'value3': 'test'}}
    >>> right = {'input': {'value': 'test2', 'value2': 'test'}}
    >>> merge(left, right, raise_on_left_missing=False)
    {'input': {'value': 'test2', 'value3': 'test', 'value2': 'test'}}

    Args:
        left:
        right:
        path:
        raise_on_left_missing:

    Returns:

    """

    for key, right_value in right.items():
        left_value = left.get(key)
        if left_value:
            if isinstance(left_value, dict) and isinstance(right_value, dict):
                merge(
                    left_value,
                    right_value,
                    path=(*path, str(key)),
                    raise_on_left_missing=raise_on_left_missing,
                )
            elif left_value != right_value:
                left[key] = right_value
        else:
            if raise_on_left_missing:
                raise ValueError(
                    f"La clé {'.'.join(map(str, (*path, key)))} n'existe pas dans le dictionnaire de gauche"
                )
            else:
                left[key] = right_value
    return left


def diff(left, right, path: tuple = (), diffs: list | None = None):
    """

    >>> left = {'input': {'value': 'test', 'value2': 'test2'}}
    >>> right = {'input': {'value': 'test2', 'value2': 'test2'}}
    >>> diff(left, right)
    {'input': {'value': 'test2'}}

    >>> left = {'input': {'value': 'test', 'inner_value': {'value': 'test'}}}
    >>> right = {'input': {'value': 'test', 'inner_value': {'value': 'test2'}}}
    >>> diff(left, right)
    {'input': {'inner_value': {'value': 'test2'}}}

    >>> left = {'input': {'value': 'test2', 'inner_value': {'value': 'test'}}}
    >>> right = {'input': {'value': 'test', 'inner_value': {'value': 'test2'}}}
    >>> diff(left, right)
    {'input': {'value': 'test', 'inner_value': {'value': 'test2'}}}

    Args:
        left:
        right:

    Returns:

    """
    if diffs is None:
        diffs = list()

    for key, right_value in right.items():
        left_value = left.get(key)
        if left_value:
            if isinstance(left_value, dict) and isinstance(right_value, dict):
                diff(left_value, right_value, (*path, key), diffs=diffs)

            elif left_value != right_value:
                diffs.append(
                    reduce(
                        lambda x, y: {y: x} if x is not None else y,
                        (*path, key)[::-1],
                        right_value,
                    )
                )
        else:
            raise ValueError(
                f"La clé {'.'.join(map(str, (*path, key)))} n'existe pas dans le dictionnaire de gauche"
            )

    diffs = reduce(lambda x, y: merge(x, y, raise_on_left_missing=False), diffs)
    return diffs
