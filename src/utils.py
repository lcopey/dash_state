from functools import reduce
from abc import ABCMeta
from typing import Any, Callable
from contextlib import contextmanager
from inspect import Signature


class ConsistencyError(Exception): ...


class InputError(Exception): ...


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
        raise InputError("\n".join(error_msg))


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
        >>> BaseState._fields
        ['input', 'value', 'object']
        >>> BaseState.__annotations__
        {'input': <class 'str'>, 'value': <class 'float'>, 'object': <class 'set'>}
        >>> BaseState._default
        {'value': 10, 'object': set()}

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

        return self_instance


class BaseState(metaclass=BaseStateMeta):
    """
    >>> class State(BaseState):
    ...     input: str
    ...     value: float = 10
    ...     object = set()
    >>> try:
    ...     State()
    ... except InputError as e:
    ...     e
    InputError('input: field required')

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
    __annotations__: dict[str, type]

    def __init__(self, *args, **kwargs):
        self.define_attributes(*args, **kwargs)
        self._watch_for_change = False
        self._parent = None
        self._name = None
        self._child: list["BaseState"] | None = None
        for field in self._fields:
            attr = getattr(self, field)
            if isinstance(attr, BaseState):
                attr.parent = self
                attr.name = field
                self.child = attr

        self.change = {}

    def _from_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            if key in self.__annotations__ or key in self._default:
                setattr(self, key, value)
            else:
                raise ValueError(f"Unexpected keyword {key}")

    def _set_default(self):
        for key, value in self._default.items():
            attr = getattr(self, key, None)
            if not attr:
                setattr(self, key, value)

    def define_attributes(self, *args, **kwargs):
        _check_required_arg(self.__annotations__, self._default, kwargs)
        # for base in bases:
        #     for key, obj in base.__dict__.items():
        #         if not _is_private_or_special(key):
        #             setattr(self, key, obj)
        if not kwargs and (args and len(args) == 1):
            raise NotImplemented()
        elif kwargs and not args:
            self._from_kwargs(kwargs)
            self._set_default()
        elif not kwargs and not args:
            self._set_default()
        else:
            raise TypeError(
                "Either one dict argument or multiple keyword arguments only"
            )

    def __repr__(self):
        attrs = ", ".join(f"{field}={getattr(self, field)!r}" for field in self._fields)
        return f"{self.__class__.__name__}({attrs})"

    @property
    def parent(self) -> "BaseState":
        """

        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         value: str
        ...
        ...     input: Input
        >>> state = State(input=State.Input(value='str'))
        >>> state.input.parent
        State(input=Input(value='str'))

        Returns:

        """
        return self._parent

    @parent.setter
    def parent(self, value: "BaseState"):
        if not self._parent:
            self._parent = value
        else:
            raise TypeError("parent attribute has already been set")

    @property
    def child(self):
        """
        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         value: str
        ...
        ...     input: Input
        >>> state = State(input=State.Input(value='str'))
        >>> state.child
        [Input(value='str')]

        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         value: str
        ...     class SecondInput(BaseState):
        ...         value: str
        ...
        ...     input: Input
        ...     second_input: SecondInput
        >>> state = State(input=State.Input(value='str'), second_input=State.SecondInput(value='str'))
        >>> state.child
        [Input(value='str'), SecondInput(value='str')]
        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         value: str
        ...
        ...     input: Input
        ...     second_input: str
        >>> state = State(input=State.Input(value=''), second_input='')
        >>> state.child
        [Input(value='')]
        """
        return self._child

    @child.setter
    def child(self, value: "BaseState"):
        if self._child:
            self._child.append(value)
        else:
            self._child = [value]

    @property
    def name(self):
        """
        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         value: str
        ...
        ...     input: Input
        >>> state = State(input=State.Input(value='str'))
        >>> state.input.name
        'input'
        """
        return self._name

    @name.setter
    def name(self, value: str):
        if not self._name:
            self._name = value
        else:
            raise TypeError("name attribute has already been set")

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
        # TODO Refaire dans __init__
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

    def set_watch_for_change(self, value: bool):
        self._watch_for_change = value
        if self.child:
            for child in self.child:
                child.set_watch_for_change(value)

    @contextmanager
    def watch_for_change(self):
        """

        >>> class State(BaseState):
        ...     class Input(BaseState):
        ...         value: str
        ...
        ...     input: Input
        ...     value: str
        >>> state = State(value='', input=State.Input(value=''))
        >>> state._watch_for_change
        False
        >>> with state.watch_for_change():
        ...     state._watch_for_change
        True
        >>> state._watch_for_change
        False
        >>> with state.watch_for_change():
        ...     state.value = 'a value'
        ...     state.change
        {'value': 'a value'}
        >>> state
        State(input=Input(value=''), value='a value')
        >>> state = State(value='', input=State.Input(value=''))
        >>> with state.watch_for_change():
        ...     state.input.value = 'a value'
        ...     state.change
        {'input': {'value': 'a value'}}

        Returns:

        """
        self.set_watch_for_change(True)
        try:
            yield
        finally:
            self.set_watch_for_change(False)

    def update_change(self, key, value):
        self.change.update({key: value})

    def __setattr__(self, key, value):
        if key == "_watch_for_change":
            super().__setattr__(key, value)
        else:
            has_diff = getattr(self, "_watch_for_change", None)
            if not has_diff or (has_diff and not self._watch_for_change):
                super().__setattr__(key, value)
            else:
                change = {key: value}
                self.update_change(key, value)
                if self.parent:
                    self.parent.update_change(self.name, change)
                super().__setattr__(key, value)

    # def __getattr__(self, item):
    #     if not self._diff:
    #         return super().__getattr__(item)


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
    ... except ConsistencyError as e:
    ...     e
    ConsistencyError("La clé input.value2 n'existe pas dans le dictionnaire de gauche")
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
                raise ConsistencyError(
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
        left (dict):
        right (dict):
        path (list):
        diffs (list):

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
            raise ConsistencyError(
                f"La clé {'.'.join(map(str, (*path, key)))} n'existe pas dans le dictionnaire de gauche"
            )

    diffs = reduce(lambda x, y: merge(x, y, raise_on_left_missing=False), diffs)
    return diffs
