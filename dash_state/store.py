"""
>>> from pydantic import BaseModel
>>> class AppState(BaseModel):
...     value: str = ''

>>> master_store = Store(id='store', state_factory=AppState)
>>> master_store
Div([Store(id='store', data={'value': ''}, storage_type='session')])
>>> master_store.input
<Input `store.data`>
>>> master_store.state
<State `store.data`>
>>> master_store.surrogates(Idx(type='surrogate').idx((Input('input', 'value'),)))
Store(id=Idx(type='surrogate', idx='70e842a92a447f0653f79bf869a2e550166134b03bc759d8d61b1a9a1a06958c'))
>>> master_store
Div([Store(id='store', data={'value': ''}, storage_type='session'),
Store(id=Idx(type='surrogate', idx='70e842a92a447f0653f79bf869a2e550166134b03bc759d8d61b1a9a1a06958c'))])
"""

from dash import (
    html,
    dcc,
    Output,
    Input,
    State,
    callback,
    clientside_callback,
    no_update,
)
from dash.dependencies import DashDependency
from pydantic import BaseModel, TypeAdapter

from typing import Literal, Callable, TypeVar, Any
from .idx import Idx
from .fast_dependencies import DccStore
from .proxy import Proxy
from .utils import filter_input, filter_state, filter_output
from functools import wraps

T = TypeVar("T", bound=BaseModel)

__all__ = ["Store", "StoreError"]


class StoreError(TypeError): ...


UPDATE_FORGOT_STATE_MSG_ERROR = """
Les fonctions décorées par Store.update doivent prendre l'état de l'application en dernier argument.

store = Store(...)
input = dcc.Input(id=...)

@store.update(Input(input, 'value'))
def callback(value, state):
    ...

"""

UPDATE_NO_RETURN_MSG_ERROR = """
Les fonctions décorées par Store.update ne doivent rien retourner et modifie l'état de l'application
en mutant directement la variable state passé en argument :

store = Store(...)
input = dcc.Input(id=...)

@store.update(Input(input, 'value')
def callback(value, state):
    state.input.value = value

"""

ON_INIT_FORGOT_STATE_MSG_ERROR = """
Les fonctions décorées par Store.on_init doivent prendre l'état de l'application en argument.

store = Store(...)
input = dcc.Input(id=...)

@store.on_init(Output(input, 'value'))
def callback(state: AppState):
    return state.value
"""

LISTEN_ON_FORGOT_INPUT_MSG_ERROR = """
Les fonctions décorées par Store.listen_on doivent prendre la valeur surveillée en premier argument.

store = Store(...)
input = dcc.Input(id=...)

@store.listen_on(store.proxy_path.value, Output(input, 'value'))
def callback(value: str):
    ...
"""


class Store(html.Div):
    def __init__(
        self,
        id: str,
        state_factory: type[T],
        data: dict | T | None = None,
        storage_type: Literal["local", "session", "memory"] = "session",
    ):
        self._state_factory = state_factory
        self._storage_type = storage_type

        if data is None:
            self._default_data = self._state_factory()
        elif isinstance(data, BaseModel):
            self._default_data = data

        if data is None:
            data = self._default_data.model_dump()

        if not isinstance(data, dict):
            raise TypeError(
                f"'data' is supposed to be a dict or {self._state_factory} instance."
            )
        self._store = DccStore(id=id, data=data, storage_type=storage_type)
        self._surrogate_stores: dict[str, dcc.Store] = dict()

        super().__init__([self._store])

    def surrogates(self, idx: Idx | str, **kwargs) -> DccStore:
        key = idx if isinstance(idx, str) else idx.immutable()
        if key not in self._surrogate_stores:
            store = DccStore(id=idx, **kwargs)
            self.children.append(store)
            self._surrogate_stores[key] = store
        else:
            store = self._surrogate_stores[key]
        return store

    @property
    def input(self):
        return self._store.input

    @property
    def state(self):
        return self._store.state

    @property
    def output(self):
        return self._store.output

    def update(self, *inputs: Input | State, **kwargs):
        prevent_initial_call = kwargs.pop("prevent_initial_call", True)

        def wrapper(func: Callable):
            @callback(
                self.output,
                *inputs,
                self.state,
                prevent_initial_call=prevent_initial_call,
                **kwargs,
            )
            def nested_callback(*args):
                args = list(args)
                # Get the state as the last argument
                # Act as clone and break any reference to the original object
                state = self._state_factory(**args.pop())
                try:
                    result = func(*args, state=state)
                    if result is not None:
                        raise StoreError(UPDATE_NO_RETURN_MSG_ERROR)
                except TypeError as e:
                    if "positional argument" in e.args[0]:
                        raise StoreError(UPDATE_FORGOT_STATE_MSG_ERROR)
                    else:
                        raise e
                return state.model_dump()

            return wraps(func)(nested_callback)

        return wrapper

    def clientside_update(
        self, clientside_function: str, *inputs: Input | State, **kwargs
    ):
        prevent_initial_call = kwargs.pop("prevent_initial_call", True)
        js_template = """
            function({signature}) {{
                // Détruit les liens entre la variable initiale et l'état passé au callback
                clone = (state) => JSON.parse(JSON.stringify(state));
                state = clone(state);
                callback = {clientside_function};
                callback({signature});
                return state;
            }}
        """
        signature = [f"arg{n}" for n in range(len(inputs))]
        signature.append("state")
        signature = ", ".join(signature)

        clientside_callback(
            js_template.format(
                clientside_function=clientside_function, signature=signature
            ),
            self.output,
            *inputs,
            self.state,
            prevent_initial_call=prevent_initial_call,
            **kwargs,
        )

    def init(self, output: Output, **kwargs):
        prevent_initial_call = kwargs.pop("prevent_initial_call", False)

        idx = Idx(type="init_store").idx((output,))
        surrogate = self.surrogates(idx, storage_type="memory", data=True)

        def wrapper(func: Callable[[T], Any]):
            @callback(
                output,
                surrogate.output,
                surrogate.input,
                self.state,
                prevent_initial_call=prevent_initial_call,
                **kwargs,
            )
            def nested_callback(on_init: bool, state: dict):
                if on_init:
                    state = self._state_factory(**state)
                    try:
                        result = func(state)
                    except TypeError as e:
                        if "positional argument" in e.args[0]:
                            raise StoreError(ON_INIT_FORGOT_STATE_MSG_ERROR)
                        else:
                            raise e
                    return result, False
                else:
                    return no_update, False

            return wraps(func)(nested_callback)

        return wrapper

    def clientside_init(self, clientside_function: str, output: Output, **kwargs):
        prevent_initial_call = kwargs.pop("prevent_initial_call", False)
        idx = Idx(type="init_store").idx((output,))
        surrogate = self.surrogates(idx, storage_type="memory", data=True)

        js_template = """
        function(on_init, state) {{
            if (on_init) {{
                callback = {clientside_function};
                result = callback(state)
                return [result, false];
            }} else {{
                return [window.dash_clientside.no_update, false]; 
            }}
        }}
        """
        clientside_callback(
            js_template.format(clientside_function=clientside_function),
            output,
            surrogate.output,
            surrogate.input,
            self.state,
            prevent_initial_call=prevent_initial_call,
            **kwargs,
        )

    @property
    def path_proxy(self) -> Proxy[T]:
        return Proxy(self._state_factory)

    def listen_on(self, path: Proxy[T], *dependencies: DashDependency, **kwargs):
        # TODO vérifier si c'est bien le comportement voulu
        prevent_initial_call = kwargs.pop("prevent_initial_call", True)
        idx = Idx(type="surrogate").idx("-".join(path))

        # TODO initialise le surrogate store avec la valeur de l'état initial ?
        # TODO prevent_initial_call est mis à True, donc on initialise.
        init = self._default_data
        for sub_path in path:
            init = getattr(init, sub_path)
        if isinstance(init, BaseModel):
            init = init.model_dump()

        surrogate_store = self.surrogates(idx, storage_type="memory", data=init)

        # Callback alimentant un surrogate store à partir de l'attribut placé dans path
        clientside_callback(
            f"""
            function(state) {{
                let result = state;
                for (const subpath of [{', '.join(map(lambda x: f'{x!r}', path))}]) {{
                    result = result[subpath];
                }}
                return result;
            }}
            """,
            surrogate_store.output,
            self.input,
            prevent_initial_call=prevent_initial_call,
        )

        outputs = filter_output(*dependencies)
        inputs = filter_input(*dependencies)
        states = filter_state(*dependencies)

        def wrapper(func):
            @callback(
                *outputs,
                surrogate_store.input,
                *inputs,
                *states,
                prevent_initial_call=prevent_initial_call,
                **kwargs,
            )
            def nested_callback(*args):
                try:
                    args = list(args)
                    state = args.pop(0)
                    # if path.model and not isinstance(state, path.model):
                    if path.model:
                        state = TypeAdapter(path.model).validate_python(state)
                    result = func(state, *args)
                except TypeError as e:
                    if "positional argument" in e.args[0]:
                        raise StoreError(LISTEN_ON_FORGOT_INPUT_MSG_ERROR)
                    else:
                        raise e
                return result

            return nested_callback

        return wrapper

    def clientside_listen_on(
        self,
        clientside_function: str,
        path: Proxy[T],
        *dependencies: DashDependency,
        **kwargs,
    ):
        prevent_initial_call = kwargs.pop("prevent_initial_call", True)
        idx = Idx(type="surrogate").idx("-".join(path))

        # TODO initialise le surrogate store avec la valeur de l'état initial ?
        # TODO prevent_initial_call est mis à True, donc on initialise.
        init = self._default_data
        for sub_path in path:
            init = getattr(init, sub_path)
        if isinstance(init, BaseModel):
            init = init.model_dump()

        surrogate_store = self.surrogates(idx, storage_type="memory", data=init)

        # Callback alimentant un surrogate store à partir de l'attribut placé dans path
        clientside_callback(
            f"""
            function(state) {{
                let result = state;
                for (const subpath of [{', '.join(map(lambda x: f'{x!r}', path))}]) {{
                    result = result[subpath];
                }}
                return result;
            }}
            """,
            surrogate_store.output,
            self.input,
            prevent_initial_call=prevent_initial_call,  # TODO vérifier si c'est bien le comportement voulu
        )

        outputs = filter_output(*dependencies)
        inputs = filter_input(*dependencies)
        states = filter_state(*dependencies)

        clientside_callback(
            clientside_function,
            *outputs,
            surrogate_store.input,
            *inputs,
            *states,
            prevent_initial_call=prevent_initial_call,
            **kwargs,
        )
