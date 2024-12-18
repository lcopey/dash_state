from dash_state import Store, DccInput
from pydantic import BaseModel
from dash import html, Dash, Output, callback
import sys
from typing import TYPE_CHECKING
from .utils import random_string

if TYPE_CHECKING:
    from dash.testing.composite import DashComposite

sys.path.append("/usr/laurent/Téléchargements/")


def make_app(clientside: bool = False, n_input: int = 1) -> Dash:
    class AppState(BaseModel):
        value: str = ""

    inputs = [DccInput(id=f"input{n}", value="") for n in range(n_input)]
    store = Store(id="store", state_factory=AppState)
    store_preview = html.Label(id="store_preview")
    layout = [html.H1("Application de base"), *inputs, store, store_preview]

    if not clientside:

        @store.update(*(input_.input for input_ in inputs))
        def on_input_change(*values: str, state: AppState):
            change = "".join(values)
            state.value = change
            return state
    else:
        signature = ", ".join([f"arg{n}" for n in range(n_input)])
        function_eval = f"''.concat({signature})"

        signature += ", state"
        store.clientside_update(
            f"({signature}) => {{"
            f"  state.value = {function_eval};"
            f"  return state;}}",
            *(input_.input for input_ in inputs),
        )

    @callback(Output(store_preview, "children"), store.input)
    def update_store_preview(state: dict):
        return f"{state!r}"

    app = Dash()
    app.layout = layout
    return app


def setup(dash_duo: "DashComposite", clientside: bool = False, n_input: int = 1):
    app = make_app(clientside=clientside, n_input=n_input)
    dash_duo.start_server(app)


def assert_store_update(
    dash_duo: "DashComposite", msg: str, expected: str, input_id: str = "input0"
):
    input_ = dash_duo.find_element(f"#{input_id}")
    input_.send_keys(msg)

    # does not work...
    # assert dash_duo.get_logs() == [], "browser console should contain no error"

    dash_duo.wait_for_text_to_equal("#store_preview", expected, timeout=4)
    assert (
        dash_duo.find_element("#store_preview").text == expected
    ), "store has not been updated"


def test_serverside_store_update(dash_duo: "DashComposite"):
    setup(dash_duo, clientside=False, n_input=1)
    msg = "test"
    expected = f"{{'value': {msg!r}}}"
    assert_store_update(dash_duo, msg, expected, input_id="input0")


def test_clientside_store_update(dash_duo: "DashComposite"):
    setup(dash_duo, clientside=True, n_input=1)
    msg = "test"
    expected = f"{{'value': {msg!r}}}"
    assert_store_update(dash_duo, msg, expected, input_id="input0")


def test_serverside_multiple_store_udpate(dash_duo: "DashComposite"):
    msg_pool = tuple(random_string() for _ in range(3))
    setup(dash_duo, clientside=False, n_input=len(msg_pool))
    messages = []
    for n, msg in enumerate(msg_pool):
        messages.append(msg)
        expected = f"{{'value': {''.join(messages)!r}}}"
        assert_store_update(dash_duo, msg, expected, f"input{n}")


def test_clientside_multiple_store_udpate(dash_duo: "DashComposite"):
    msg_pool = tuple(random_string() for _ in range(3))
    setup(dash_duo, clientside=True, n_input=len(msg_pool))
    messages = []
    for n, msg in enumerate(msg_pool):
        messages.append(msg)
        expected = f"{{'value': {''.join(messages)!r}}}"
        assert_store_update(dash_duo, msg, expected, f"input{n}")
