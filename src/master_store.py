from dash import html, dcc, callback, ALL, Input, Output, State, callback_context
import sys
from typing import Any
from dataclasses import dataclass
from copy import deepcopy


class StoreError(TypeError): ...


FORGOT_STATE_MSG_ERROR = """
Les fonctions décorés par ReduxStore.udate doivent prendre l'état de l'application en dernier argument.

store = ReduxStore(...)
input = dcc.Input(id=...)

@store.update(Input(input, 'value')
def callback(value, state):
    ...

"""


def unpack(id):
    if isinstance(id, dict):
        id = [part for part in id.values() if isinstance(part, str)]
        return "_".join(id)
    return id


def input_hash(*args: Input | State) -> str:
    inputs = [
        ".".join((unpack(arg.component_id), arg.component_property)) for arg in args
    ]
    _hash = hash(tuple(inputs))
    _hash += sys.maxsize + 1
    return hex(_hash)[2:]


@dataclass
class IdComposer:
    type: str

    def _idx(self, value: Any):
        return {"type": self.type, "idx": value}

    def idx(self, value: Any):
        return self._idx(value)

    def all(self):
        return self._idx(ALL)


def triggered_index(values: dict):
    callback_context.triggered_id['idx']


class ReduxStore(html.Div):
    """

    >>> Redux = ReduxStore('store', model={})
    >>> Redux
    Div([Store(id='store', storage_type='session')])

    """

    def __init__(self, id: str, model, **kwargs):
        storage_type = kwargs.pop("storage_type", "session")
        self.state_model = model
        self.master_store = dcc.Store(id=id, storage_type=storage_type, **kwargs)

        self._surrogate_store_match = IdComposer(type=f"{id}_ip")
        self._surrogate_stores = dict()
        super().__init__([self.master_store])

        @callback(
            Output(self.master_store, "data"),
            Input(self._surrogate_store_match.idx(ALL), "data"),
            prevent_initial_call=True,
        )
        def update_store(surrogate_state):
            callback_context.triggered_id["idx"]

    def _surrogate_input_store(self, *inputs: Input | State) -> dcc.Store:
        id = input_hash(*inputs)
        if id not in self._surrogate_stores:
            # Initialise les valeurs avec la forme des données du master store
            store = dcc.Store(
                id=self._surrogate_store_match.idx(id),
                data=self.master_store.data,
                storage_type="memory",
            )
            self.children.append(store)
            self._surrogate_stores[id] = store
        return self._surrogate_stores[id]

    def update(self, *inputs: Input | State, **callback_kwargs):
        surrogate_store = self._surrogate_input_store(*inputs)

        def wrapper(func):
            prevent_initial_call = callback_kwargs.pop("prevent_initial_call", True)

            @callback(
                Output(surrogate_store, "data"),
                *inputs,
                State(self.master_store, "data"),
                prevent_initial_call=prevent_initial_call,
                **callback_kwargs,
            )
            def _proxy(*args):
                args = list(args)
                args[-1] = deepcopy(args[-1])
                args[-1] = self.state_model(**args[-1])

                try:
                    result = func(*args)
                except TypeError as e:
                    if "positional argument" in e.args[0]:
                        raise StoreError(FORGOT_STATE_MSG_ERROR)
                    else:
                        raise e
                return result

        return wrapper
