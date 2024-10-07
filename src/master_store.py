from dash import (
    html,
    dcc,
    callback,
    ALL,
    Input,
    Output,
    State,
    callback_context,
    no_update,
    clientside_callback,
)
import sys
from typing import Any
from typing import Literal, Hashable
from .state import BaseState

from dash.exceptions import PreventUpdate


class StoreError(TypeError): ...


FORGOT_STATE_MSG_ERROR = """
Les fonctions décorées par ReduxStore.update doivent prendre l'état de l'application en dernier argument.

store = ReduxStore(...)
input = dcc.Input(id=...)

@store.update(Input(input, 'value')
def callback(value, state):
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


class IdComposer:
    """

    >>> IdComposer(type='store')
    IdComposer(type='store')
    >>> IdComposer(type='store').bind(mode='callback')
    IdComposer(type='store', mode='callback')
    >>> IdComposer(type='store').idx('value')
    {'type': 'store', 'idx': 'value'}
    >>> IdComposer(type='store').idx(ALL)
    {'type': 'store', 'idx': <ALL>}
    >>> IdComposer(type='store').bind(mode='callback').idx(ALL)
    {'type': 'store', 'mode': 'callback', 'idx': <ALL>}

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
        return IdComposer(**self.kwargs, **kwargs)

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

    >>> class State(BaseState):
    ...     input: str
    >>> Redux = ReduxStore('store', state_factory=State)
    >>> Redux
    Div([Store(id='store', data={'input': None}, storage_type='session')])
    >>> Redux._surrogate_stores_match
    IdComposer(type='surrogate_store')

    """

    def __init__(
        self,
        id: str,
        state_factory: type[BaseState],
        data: BaseState | None = None,
        **kwargs,
    ):
        self._storage_type = kwargs.pop("storage_type", "session")
        self._state_factory = state_factory
        data = data or state_factory.default().as_dict()
        self._master_store = dcc.Store(
            id=id, storage_type=self._storage_type, data=data, **kwargs
        )

        self._surrogate_stores_match = IdComposer(type=f"surrogate_{id}")
        self._surrogate_stores: dict[tuple[Hashable, str], dcc.Store] = dict()

        class _Proxy:
            as_input = Input(self._master_store, "data")
            as_state = State(self._master_store, "data")

        self.store = _Proxy

        super().__init__([self._master_store])

        # @callback(
        #     Output(self._master_store, "data"),
        #     Input(self._surrogate_stores_match.bind(mode="callback").idx(ALL), "data"),
        #     self.store.as_state,
        #     prevent_initial_call=True,
        # )
        # def update_master_store(surrogate_state, current_state: dict):
        #     print("update_master_store")
        #     index = trigger_index()
        #     if index is not None:
        #         merged = self._state_factory.from_dict(current_state).merge(
        #             surrogate_state[index]
        #         )
        #
        #         return merged.as_dict()
        #     raise PreventUpdate()
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
                console.log('update_master_store', results);
                return results;
            }""",
            Output(self._master_store, "data"),
            Input(self._surrogate_stores_match.bind(mode="callback").idx(ALL), "data"),
            self.store.as_state,
            prevent_initial_call=True,
        )

    def _surrogate_input_store(
        self,
        *inputs: Input | State,
        mode: Literal["callback", "initial", "on_init"],
        default: Any | None = None,
    ) -> dcc.Store:
        """

        Args:
            *inputs: Dash Input ou State servant à définir l'id du futur surrogate store
            mode Literal["callback", "initial", "initial_state"]:
                - callback: définit un Store qui prendra prendra les changements initiés par les inputs
                    Dans ce mode, la valeur par défaut n'est pas utilisé et le master_store est utilisé
                    # TODO prendre un simple dictionnaire à la place ?
                - initial: définit un Store qui stockera les valeurs à l'initial du composant surveillé
                    Dans ce mode, une valeur par défaut peut être fourni
                - on_init: définit un Store dont la seule finalité sera de dire si le premier callback
                    a déjà été executé ou non.
            default:

        Returns:

        """
        idx = _input_hash(*inputs)
        if idx not in self._surrogate_stores:
            # Initialise les valeurs avec la forme des données du master store
            store_id = self._surrogate_stores_match.bind(mode=mode).idx(idx)
            if mode == "callback":
                storage_type = self._storage_type
                initial_data = self._master_store.data
            elif mode == "initial":
                storage_type = self._storage_type
                initial_data = default
            elif mode == "on_init":
                storage_type = "memory"
                initial_data = True
            else:
                raise ValueError(f"Unknow mode : {mode}")

            store = dcc.Store(
                id=store_id,
                data=initial_data,
                storage_type=storage_type,
            )
            self.children.append(store)
            self._surrogate_stores[(idx, mode)] = store
        return self._surrogate_stores[(idx, mode)]

    def store_initial(
        self,
        component_id,
        component_property,
        on: str | None = None,
        default: Any | None = None,
        **callback_kwargs,
    ):
        if on:
            input_ = Input(component_id=component_id, component_property=on)
            state_data = State(
                component_id=component_id, component_property=component_property
            )
        else:
            input_ = Input(
                component_id=component_id, component_property=component_property
            )
            state_data = None

        value_surrogate_store = self._surrogate_input_store(
            input_, mode="initial", default=default
        )
        on_init_surrogate_store = self._surrogate_input_store(input_, mode="on_init")
        inputs = (
            input_,
            Input(value_surrogate_store, "data"),
            State(on_init_surrogate_store, "data"),
        )
        if state_data:
            inputs = (*inputs, state_data)

        prevent_initial_call = callback_kwargs.pop("prevent_initial_call", False)

        # @callback(
        #     Output(value_surrogate_store, "data"),
        #     Output(on_init_surrogate_store, "data"),
        #     Output(component_id, component_property),
        #     *inputs,
        #     prevent_initial_call=prevent_initial_call,
        # )
        # def _proxy(*args):
        #     *args, store, on_init = args
        #     print("store_initial", args, store, on_init)
        #     if on_init:
        #         return no_update, False, store
        #     else:
        #         return args[0], False, no_update

        clientside_callback(
            """function(...args) {
                let datas, store, on_init;
                let no_update = window.dash_clientside.no_update
                if (args.length === 3) {
                    [datas, store, on_init] = args; 
                } else {
                    [_, store, on_init, datas] = args;
                }
                if (on_init) {
                    return [no_update, false, store]; 
                } else {
                    return [datas, false, no_update];
                }
            }""",
            Output(value_surrogate_store, "data"),
            Output(on_init_surrogate_store, "data"),
            Output(component_id, component_property),
            *inputs,
            prevent_initial_call=prevent_initial_call,
        )

    def update(self, *inputs: Input | State, **callback_kwargs):
        surrogate_store = self._surrogate_input_store(*inputs, mode="callback")

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
                print("update surrogate")
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
