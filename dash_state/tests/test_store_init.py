from dash_state import Store, DccInput
from pydantic import BaseModel
from dash import html, Dash, Output, callback
import sys
from typing import TYPE_CHECKING
import time

from .utils import random_string

if TYPE_CHECKING:
    from dash.testing.composite import DashComposite

sys.path.append("/usr/laurent/Téléchargements/")


def make_app(clientside: bool = False):
    class AppState(BaseModel):
        value: str = ""

    input_ = DccInput(id=f"input", value="")
    store = Store(id="store", state_factory=AppState)
    store_preview = html.Label(id="store_preview")
    layout = [html.H1("Application de base"), input_, store, store_preview]

    @store.update(input_.input)
    def on_input_change(value: str, state: AppState):
        state.value = value

    if not clientside:

        @store.init(input_.output)
        def init_input(state: AppState):
            return state.value
    else:
        store.clientside_init("state => state.value", input_.output)

    @callback(
        Output(store_preview, "children"), store.input, prevent_initial_callback=True
    )
    def update_store_preview(state):
        return f"{state!r}"

    app = Dash()
    app.layout = layout
    return app


def setup(dash_duo: "DashComposite", clientside: bool, msg: str):
    app = make_app(clientside=clientside)
    dash_duo.start_server(app)
    dash_duo.clear_storage()
    # attends que l'ensemble des callbacks soient résolus une première fois.
    dash_duo.wait_for_text_to_equal("#store_preview", "{'value': ''}")
    time.sleep(0.5)
    input_ = dash_duo.find_element(f"#input")
    input_.send_keys(msg)


def test_store_serverside_init(dash_duo: "DashComposite"):
    msg = random_string()
    expected = f"{{'value': {msg!r}}}"
    setup(dash_duo, clientside=False, msg=msg)
    dash_duo.driver.refresh()
    dash_duo.wait_for_text_to_equal("#store_preview", expected, timeout=4)
    assert dash_duo.find_element("#store_preview").text == expected


def test_store_clientside_init(dash_duo: "DashComposite"):
    msg = random_string()
    expected = f"{{'value': {msg!r}}}"
    setup(dash_duo, clientside=True, msg=msg)
    dash_duo.driver.refresh()
    dash_duo.wait_for_text_to_equal("#store_preview", expected, timeout=4)
    assert dash_duo.find_element("#store_preview").text == expected
