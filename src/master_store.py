from dash import html, dcc, callback, ALL, Input, Output, State, callback_context
import sys
from typing import Any
from dataclasses import dataclass
from copy import deepcopy

from dash.exceptions import PreventUpdate


class StoreError(TypeError): ...


FORGOT_STATE_MSG_ERROR = """
Les fonctions décorés par ReduxStore.udate doivent prendre l'état de l'application en dernier argument.

store = ReduxStore(...)
input = dcc.Input(id=...)

@store.update(Input(input, 'value')
def callback(value, state):
    ...

"""


@dataclass
class IdComposer:
    type: str

    def _idx(self, value: Any):
        return {"type": self.type, "idx": value}

    def idx(self, value: Any):
        return self._idx(value)

    def all(self):
        return self._idx(ALL)


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


def _input_hash(*args: Input | State) -> str:
    inputs = [
        ".".join((_unpack(arg.component_id), arg.component_property)) for arg in args
    ]
    _hash = hash(tuple(inputs))
    _hash += sys.maxsize + 1
    return hex(_hash)[2:]


class ReduxStore(html.Div):
    """

    >>> Redux = ReduxStore('store', data={})
    >>> Redux
    Div([Store(id='store', data={}, storage_type='session')])
    >>> Redux._surrogate_store_match
    IdComposer(type='store_ip')

    """

    def __init__(self, id: str, data: Any, **kwargs):
        storage_type = kwargs.pop("storage_type", "session")
        # self.state_model = model
        self._master_store = dcc.Store(
            id=id, storage_type=storage_type, data=data, **kwargs
        )

        self._surrogate_store_match = IdComposer(type=f"{id}_ip")
        self._surrogate_stores = dict()

        class _Proxy:
            input = Input(self._master_store, "data")
            output = Output(self._master_store, "data")
            state = State(self._master_store, "data")

        self.store = _Proxy

        super().__init__([self._master_store])

        @callback(
            Output(self._master_store, "data"),
            Input(self._surrogate_store_match.idx(ALL), "data"),
            prevent_initial_call=True,
        )
        def update_store(surrogate_state):
            index = trigger_index()
            if index is not None:
                state = surrogate_state[index]
                print("Update master store", state)
                return state
            raise PreventUpdate()

    def _surrogate_input_store(self, *inputs: Input | State) -> dcc.Store:
        idx = _input_hash(*inputs)
        if idx not in self._surrogate_stores:
            # Initialise les valeurs avec la forme des données du master store
            store = dcc.Store(
                id=self._surrogate_store_match.idx(idx),
                data=self._master_store.data,
                storage_type="memory",
            )
            self.children.append(store)
            self._surrogate_stores[idx] = store
        return self._surrogate_stores[idx]

    # def initial_update(self, input_: Input, **callback_kwargs):
    #     def wrapper(func):
    #         @callback(
    #             Output(self.master_store, 'data'),
    #             input_,
    #             State(self.master_store, 'data')
    #         )
    #         def _proxy():
    #             pass

    def update(self, *inputs: Input | State, **callback_kwargs):
        surrogate_store = self._surrogate_input_store(*inputs)

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
                state = deepcopy(args.pop())
                print("Update")
                print("State avant", state)
                # args[-1] = deepcopy(args[-1])
                # args[-1] = self.state_model(**args[-1])

                try:
                    result = func(*args, state)
                    print("State après", state)
                except TypeError as e:
                    if "positional argument" in e.args[0]:
                        raise StoreError(FORGOT_STATE_MSG_ERROR)
                    else:
                        raise e
                return result

        return wrapper
