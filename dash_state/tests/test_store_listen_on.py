from dash_state import Store, DccInput, DccLabel
from pydantic import BaseModel
from dash import html, Dash
import sys
from typing import TYPE_CHECKING

from .utils import random_string

if TYPE_CHECKING:
    from dash.testing.composite import DashComposite

sys.path.append("/usr/laurent/Téléchargements/")


def make_app(clientside: bool = False, nested: bool = False):
    if nested:

        class InputState(BaseModel):
            value: str = ""

        class AppState(BaseModel):
            input_: InputState = InputState()
    else:

        class AppState(BaseModel):
            value: str = ""

    input_ = DccInput(id=f"input", value="")
    store = Store(id="store", state_factory=AppState)
    store_preview = DccLabel(id="store_preview")
    layout = [html.H1("Application de base"), input_, store, store_preview]

    @store.update(input_.input)
    def on_input_change(value: str, state: AppState):
        if nested:
            state.input_.value = value
        else:
            state.value = value

    if clientside and not nested:
        store.clientside_listen_on(
            "input => input", store.path_proxy.value, store_preview.output
        )
    elif not clientside and not nested:

        @store.listen_on(store.path_proxy.value, store_preview.output)
        def watch_for_change_in_value(input_: str):
            return input_
    elif clientside and nested:
        store.clientside_listen_on(
            "input => input.value", store.path_proxy.input_, store_preview.output
        )
    else:

        @store.listen_on(store.path_proxy.input_, store_preview.output)
        def watch_for_change_in_value(input_: "InputState"):
            return input_.value

    app = Dash()
    app.layout = layout
    return app


def setup(dash_duo: "DashComposite", clientside: bool, nested: bool, msg: str):
    app = make_app(clientside=clientside, nested=nested)
    dash_duo.start_server(app)

    input_ = dash_duo.find_element(f"#input")
    input_.send_keys(msg)


def test_serverside_flat_listen_on(dash_duo: "DashComposite"):
    msg = random_string()
    setup(dash_duo, clientside=False, nested=False, msg=msg)
    expected = msg
    dash_duo.wait_for_text_to_equal("#store_preview", expected, timeout=4)


def test_serverside_nested_listen_on(dash_duo: "DashComposite"):
    msg = random_string()
    setup(dash_duo, clientside=False, nested=True, msg=msg)
    expected = msg
    dash_duo.wait_for_text_to_equal("#store_preview", expected, timeout=4)


def test_clientside_flat_listen_on(dash_duo: "DashComposite"):
    msg = random_string()
    setup(dash_duo, clientside=True, nested=False, msg=msg)
    expected = msg
    dash_duo.wait_for_text_to_equal("#store_preview", expected, timeout=4)


def test_clientside_nested_listen_on(dash_duo: "DashComposite"):
    msg = random_string()
    setup(dash_duo, clientside=True, nested=True, msg=msg)
    expected = msg
    dash_duo.wait_for_text_to_equal("#store_preview", expected, timeout=4)
