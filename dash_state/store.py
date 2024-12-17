"""
>>> class AppState(BaseState):
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

from dash import html, dcc, Output, Input, State, callback, clientside_callback
from dash.dependencies import DashDependency
from dash_state.base_state import BaseState

from typing import Literal, Callable, TypeVar
from .idx import Idx
from .fast_dependencies import DccStore
from functools import wraps

T = TypeVar("T", bound=BaseState)

__all__ = ["Store", "StoreError"]


class StoreError(TypeError): ...


FORGOT_STATE_MSG_ERROR = """
Les fonctions décorées par Store.update doivent prendre l'état de l'application en dernier argument.

store = Store(...)
input = dcc.Input(id=...)

@store.update(Input(input, 'value'))
def callback(value, state):
    ...

"""

NO_RETURN_MSG_ERROR = """
Les fonctions décorées par Store.update ne doivent rien retourner et modifie l'état de l'application
en mutant directement la variable state passé en argument :

store = Store(...)
input = dcc.Input(id=...)

@store.update(Input(input, 'value')
def callback(value, state):
    state.input.value = value

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
            data = self._state_factory().to_dict()
        elif isinstance(data, BaseState):
            data = data.to_dict()

        if not isinstance(data, dict):
            raise TypeError(
                f"'data' is supposed to be a dict or {self._state_factory} instance."
            )
        self._store = DccStore(id=id, data=data, storage_type=storage_type)
        self._surrogate_stores: dict[str, dcc.Store] = dict()
        # self._surrogate_stores_idx = Idx(type='surrogate')

        super().__init__([self._store])

    def surrogates(self, idx: Idx | str) -> dcc.Store:
        key = idx if isinstance(idx, str) else idx.immutable()
        if key not in self._surrogate_stores:
            store = dcc.Store(id=idx)
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
            def _(*args):
                args = list(args)
                # Act as clone and break any reference to the original object
                state = self._state_factory.from_dict(args.pop())
                try:
                    result = func(*args, state=state)
                    if result is not None:
                        raise StoreError(NO_RETURN_MSG_ERROR)
                except TypeError as e:
                    if "positional argument" in e.args[0]:
                        raise StoreError(FORGOT_STATE_MSG_ERROR)
                    else:
                        raise e
                return state.to_dict()

        return wrapper

    def update_clientside(
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
        )

    def on_init(self, dependency: DashDependency, **kwargs):
        prevent_initial_call = kwargs.pop("prevent_initial_call", False)

        idx = Idx(type="init_store").idx((dependency,))
        output = Output(dependency.component_id, dependency.component_property)

        def wrapper(func):
            pass

        return wrapper
