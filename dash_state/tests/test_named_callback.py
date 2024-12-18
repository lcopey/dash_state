import time

from dash_state.utils import callback
from dash import Dash
from dash_state.fast_dependencies import DccInput, DccLabel, DccButton
from typing import TYPE_CHECKING
import sys

sys.path.append("/usr/laurent/Téléchargements/")

if TYPE_CHECKING:
    from dash.testing.composite import DashComposite


def make_app():
    input_ = DccInput(id="input", value="")
    label = DccLabel(id="label", children="")
    button = DccButton(id="button")

    app = Dash()
    app.layout = [input_, label, button]

    @callback(
        target=label.output,
        input=input_.state,
        n_clicks=button.input.n_clicks,
        prevent_initial_call=True,
    )
    def on_click(input: str, n_clicks: int):
        return {"target": f"{n_clicks} {input}"}

    return app


def test_named_callback(dash_duo: "DashComposite"):
    app = make_app()
    dash_duo.start_server(app)
    input_ = dash_duo.find_element("#input")
    input_.send_keys("test")
    button = dash_duo.find_element("#button")
    button.click()
    dash_duo.wait_for_text_to_equal("#label", "1 test")
    assert dash_duo.find_element("#label").text == "1 test"
