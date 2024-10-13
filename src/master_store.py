from dash import (
    html,
    dcc,
    callback,
    ALL,
    MATCH,
    Input,
    Output,
    State,
    callback_context,
    clientside_callback,
)
from typing import Any, Literal, Hashable, Protocol
from hashlib import sha256
import orjson
from dataclasses import dataclass
from enum import StrEnum, auto

from dash_prefix import component_id

from .state import BaseState
from .observer import Observer, Proxy


class ComponentT(Protocol):
    component_id: str
    component_property: str


@dataclass
class Component:
    component_id: str
    component_property: str


class StoreError(TypeError): ...


FORGOT_STATE_MSG_ERROR = """
Les fonctions décorées par ReduxStore.update doivent prendre l'état de l'application en dernier argument.

store = ReduxStore(...)
input = dcc.Input(id=...)

@store.update(Input(input, 'value'))
def callback(value, state):
    ...

"""

FORGOT_STORE_MSG_ERROR = """
Les fonctions décorés par ReduxStore.on_change_of doivent prendre la valeur observée en dernier argument.

store = ReduxStore(...)
input = dcc.Input(id=...)
widget = dcc.SomeWidget(id=...)

@store.on_change_of(store.on.input.value, Input(...), ..., Output(widget, 'children'))
def callback(..., value_changed):
    ...

"""

NO_RETURN_MSG_ERROR = """
Les fonctions décorées par ReduxStore.update ne doivent rien retourner et modifie l'état de l'application
en mutant directement la variable state passé en argument :

store = ReduxStore(...)
input = dcc.Input(id=...)

@store.update(Input(input, 'value')
def callback(value, state):
    state.input.value = value

"""


class Ids:
    """

    >>> Ids(type='store')
    Ids(type='store')
    >>> Ids(type='store').bind(mode='callback')
    Ids(type='store', mode='callback')
    >>> Ids(type='store').idx('value')
    {'type': 'store', 'idx': 'value'}
    >>> Ids(type='store').idx(ALL)
    {'type': 'store', 'idx': <ALL>}
    >>> Ids(type='store').bind(mode='callback').idx(ALL)
    {'type': 'store', 'mode': 'callback', 'idx': <ALL>}
    >>> Ids(type='store').idx(MATCH)
    {'type': 'store', 'idx': <MATCH>}

    """

    def __init__(self, **kwargs):
        assert (
            "idx" not in kwargs
        ), "idx is reserved keyword and should not be in kwargs"
        self.kwargs = kwargs

    def __repr__(self):
        attrs = ", ".join(f"{key}={value!r}" for key, value in self.kwargs.items())
        return f"{self.__class__.__name__}({attrs})"

    def _idx(self, value: Any):
        return {**self.kwargs, "idx": value}

    def bind(self, **kwargs):
        return Ids(**self.kwargs, **kwargs)

    def idx(self, value: Any):
        return self._idx(value)

    def all(self):
        return self._idx(ALL)

    def match(self):
        return self._idx(MATCH)


def trigger_index(idx_field="idx") -> int | None:
    """Return the index of the match ALL field that triggered the callback

    Args:
        idx_field (str, optional): The MATCH field to test Defaults to 'idx'.

    Returns:
        Union[int, None]: Returns the index or None
    """
    ctx = callback_context
    if ctx.triggered and idx_field in ctx.triggered_id:
        triggered_idx = ctx.triggered_id[idx_field]
        for index, input in enumerate(ctx.inputs_list[0]):
            if triggered_idx == input["id"][idx_field]:
                return index
    else:
        return None


def _unpack(idx: dict | str) -> str:
    """
    >>> _unpack('str')
    'str'
    >>> _unpack({'idx': ALL, 'type': 'base'})
    'base'
    >>> _unpack({'type': 'base', 'subtype': 'input'})
    'base_input'

    Args:
        idx:

    Returns:

    """
    if isinstance(idx, dict):
        idx = [part for part in idx.values() if isinstance(part, str)]
        return "_".join(idx)
    return idx


def _input_hash(*args: ComponentT) -> Hashable:
    """
    >>> _input_hash(Input('input', 'value'))
    '70e842a92a447f0653f79bf869a2e550166134b03bc759d8d61b1a9a1a06958c'
    >>> _input_hash(Input('input', 'value'), State('store', 'data'))
    'cb61779d9e17f4ea19e28ecbf9d8859626c981fb7ad8da75f158810529e109c5'
    >>> _input_hash(Component('input', 'value'))
    '70e842a92a447f0653f79bf869a2e550166134b03bc759d8d61b1a9a1a06958c'

    Args:
        *args:

    Returns:

    """
    inputs = [
        ".".join((_unpack(arg.component_id), arg.component_property)) for arg in args
    ]
    hash_ = sha256(orjson.dumps(tuple(inputs))).hexdigest()
    return hash_


class StoreMode(StrEnum):
    CALLBACK = auto()
    INITIAL = auto()
    ON_INIT = auto()
    ON_STORE_CHANGE = auto()


MemoryT = Literal["memory", "session", "local"]


class StoreIndex(dict):
    """
    >>> stores = StoreIndex()
    >>> stores[{'type': 'store', 'idx': 'value'}] = 'value'
    >>> stores
    {b'{"type":"store","idx":"value"}': 'value'}
    >>> b'{"type":"store","idx":"value"}' in stores
    True
    >>> {"type":"store","idx":"value"} in stores
    True
    >>> {"type":"store","idx":"another value"} not in stores
    True
    """

    @staticmethod
    def _process_key(key):
        if isinstance(key, dict):
            key = orjson.dumps(key)
        return key

    def __getitem__(self, item):
        item = self._process_key(item)
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        key = self._process_key(key)
        super().__setitem__(key, value)

    def __contains__(self, item):
        key = self._process_key(item)
        return super().__contains__(key)


class ReduxStore(html.Div):
    """

    >>> class State(BaseState):
    ...     input: str
    >>> Redux = ReduxStore('store', state_factory=State)
    >>> Redux
    Div([Store(id='store', data={'input': None}, storage_type='session')])

    """

    def __init__(
        self,
        id: str,
        state_factory: type[BaseState],
        data: BaseState | None = None,
        **kwargs,
    ):
        self._store_id = id
        self._storage_type = kwargs.pop("storage_type", "session")
        self._state_factory = state_factory
        data = data or state_factory.default().as_dict()
        self._master_store = dcc.Store(
            id=id, storage_type=self._storage_type, data=data, **kwargs
        )

        self._surrogate_store_ids = lambda mode, idx: Ids(
            type=f"surrogate_{id}", mode=mode
        ).idx(idx)
        self._surrogate_stores = StoreIndex()
        self._observer = Observer(state_factory)

        class _Proxy:
            as_input = Input(self._master_store, "data")
            as_state = State(self._master_store, "data")

        self.store = _Proxy

        super().__init__([self._master_store])

        # Merge des deux objets en js
        clientside_callback(
            """
            function(surrogate_state, current_state) {
                function trigger_index() {
                    let context = window.dash_clientside.callback_context;
                    let triggered = context.inputs_list[0].map(
                        (item) => item.id.idx
                    )
                    let triggered_id = context.triggered_id.idx;
                    return triggered.indexOf(triggered_id);
                }
                function isObject(item) {
                    return (item && typeof item == 'object' && !Array.isArray(item));
                }
                function merge(target, ...sources) {
                    if (!sources.length) return target;
                    const source = sources.shift();
                    
                    if (isObject(target) && isObject(source)) {
                        for (const key in source) {
                            if (!isObject(source[key])) {
                                if (!target[key]) Object.assign(target, { [key]: {} });
                                merge(target[key], source[key]);
                            } else {
                                Object.assign(target, { [key]: source[key] });
                            }
                        }
                    }
                    return merge(target, ...sources);
                }
                
                let index = trigger_index();
                let results = merge(current_state, surrogate_state[index]);
                return results;
            }""",
            Output(self._master_store, "data"),
            Input(self._surrogate_store_ids(mode="callback", idx=ALL), "data"),
            self.store.as_state,
            prevent_initial_call=True,
        )

    @property
    def store_id(self):
        return self._store_id

    def _register_store(self, store_id: dict, initial_data: Any, storage_type: MemoryT):
        store = dcc.Store(id=store_id, data=initial_data, storage_type=storage_type)
        self.children.append(store)
        self._surrogate_stores[store_id] = store

    def _get_surrogate_callback_store(self, *inputs: ComponentT):
        idx = _input_hash(*inputs)
        store_id = self._surrogate_store_ids(mode=StoreMode.CALLBACK, idx=idx)
        if store_id not in self._surrogate_stores:
            self._register_store(store_id, self._master_store.data, self._storage_type)
        return self._surrogate_stores[store_id]

    def _get_surrogate_initial_stores(self, *inputs: ComponentT, default):
        idx = _input_hash(*inputs)
        initial_id = self._surrogate_store_ids(mode=StoreMode.INITIAL, idx=idx)
        if initial_id not in self._surrogate_stores:
            self._register_store(initial_id, default, self._storage_type)
        on_init_id = self._surrogate_store_ids(mode=StoreMode.ON_INIT, idx=idx)
        if on_init_id not in self._surrogate_stores:
            self._register_store(on_init_id, True, "memory")

        return self._surrogate_stores[initial_id], self._surrogate_stores[on_init_id]

    def store_initial(
        self,
        component_id: str,
        component_property: str,
        on: str | None = None,
        default: Any | None = None,
        **callback_kwargs,
    ):
        # TODO Utilise modified_timestamp à la place ?
        # https://dash.plotly.com/dash-core-components/store
        if on:
            input_ = Input(component_id=component_id, component_property=on)
            state_data = State(
                component_id=component_id, component_property=component_property
            )
            callback_fragment = """
            [_, store, on_init, datas] = args;
            """
        else:
            input_ = Input(
                component_id=component_id, component_property=component_property
            )
            state_data = None
            callback_fragment = """
            [store, on_init, datas] = args;
            """

        value_surrogate_store, on_init_surrogate_store = (
            self._get_surrogate_initial_stores(input_, default=default)
        )
        inputs = (
            input_,
            Input(value_surrogate_store, "data"),
            State(on_init_surrogate_store, "data"),
        )
        if state_data:
            inputs = (*inputs, state_data)

        prevent_initial_call = callback_kwargs.pop("prevent_initial_call", False)

        clientside_callback(
            f"""function(...args) {{
                let no_update = window.dash_clientside.no_update;
                let datas, store, on_init;
                {callback_fragment}
                if (on_init) {{
                    return [no_update, false, store];
                }} else {{
                    return [datas, false, no_update];
                }}
            }}""",
            Output(value_surrogate_store, "data"),
            Output(on_init_surrogate_store, "data"),
            Output(component_id, component_property),
            *inputs,
            prevent_initial_call=prevent_initial_call,
        )

    def update(self, *inputs: Input | State, **callback_kwargs):
        surrogate_store = self._get_surrogate_callback_store(*inputs)

        def wrapper(func):
            prevent_initial_call = callback_kwargs.pop("prevent_initial_call", True)

            @callback(
                Output(surrogate_store, "data"),
                *inputs,
                State(self._master_store, "data"),
                prevent_initial_call=prevent_initial_call,
                **callback_kwargs,
            )
            def _proxy(*args):
                args = list(args)
                state = self._state_factory.from_dict(args.pop())

                try:
                    with state.watch_for_change() as new_state:
                        result: Any = func(*args, new_state)
                        if result is not None:
                            raise StoreError(NO_RETURN_MSG_ERROR)

                except TypeError as e:
                    if "positional argument" in e.args[0]:
                        raise StoreError(FORGOT_STATE_MSG_ERROR)
                    else:
                        raise e
                return state.change

        return wrapper

    def _get_on_change_surrogate_store(self, path: str):
        idx = _input_hash(Component(self._store_id, path))
        store_id = self._surrogate_store_ids(mode=StoreMode.ON_STORE_CHANGE, idx=idx)
        if store_id not in self._surrogate_stores:
            self._register_store(store_id, self._master_store.data, "memory")
        return self._surrogate_stores[store_id]

    def on_change_of(
        self,
        on: Proxy[BaseState],
        *components: Input | State | Output,
        **callback_kwargs,
    ):
        path = ".".join(on._path)
        store = self._get_on_change_surrogate_store(path)

        # Ajoute un store qui est mis à jour lors d'une mise à jour du store global
        clientside_callback(
            f"""
            function(state) {{
                const path = '{path}';
                
                const getValue = (obj, path) => {{
                  return path.split('.').reduce((acc, part) => {{
                    return acc && acc[part]; // Vérifie que acc n'est pas undefined
                  }}, obj);
                }};
                
                const value = getValue(state, path);
                console.log(value);
                return value;
            }}
            """,
            Output(store, "data"),
            self.store.as_input,
        )

        outputs = [
            component for component in components if isinstance(component, Output)
        ]
        inputs = [
            component for component in components if not isinstance(component, Output)
        ]

        inputs = [*inputs, Input(store, "data")]

        def wrapper(func):
            prevent_initial_call = callback_kwargs.pop("prevent_initial_call", True)

            @callback(*outputs, *inputs, prevent_initial_call=prevent_initial_call)
            def _proxy(*args):
                if len(args) < len(inputs):
                    raise StoreError(FORGOT_STORE_MSG_ERROR)
                result = func(*args)
                return result

        return wrapper

    @property
    def on(self):
        return self._observer.on
