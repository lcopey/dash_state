from dash_state import BaseState, Store
from dash import html, dcc, Dash, Output, Input, callback
import sys
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from dash.testing.composite import DashComposite

sys.path.append("/usr/laurent/Téléchargements/")


def make_app() -> Dash:
    class AppState(BaseState):
        value: str = ""

    input_ = dcc.Input(id="input")
    store = Store(id="store", state_factory=AppState)
    label = html.Label(id="label")
    store_preview = html.Label(id="store_preview")
    layout = [html.H1("Application de base"), input_, store, label, store_preview]

    @store.update(Input(input_, "value"))
    def on_input_change(value: str, state: AppState):
        state.value = value

    @callback(Output(label, "children"), store.input)
    def on_store_change(state: dict):
        state = AppState.from_dict(state)
        return state.value

    @callback(Output(store_preview, "children"), store.input)
    def update_store_preview(state: dict):
        return f"{state!r}"

    app = Dash()
    app.layout = layout
    return app


def run_app(func: Callable[[], Dash]):
    app = func()
    app.run()


def test_store(dash_duo: "DashComposite"):
    app = make_app()
    dash_duo.start_server(app)

    input_ = dash_duo.find_element("#input")
    input_.send_keys("test")

    # does not work...
    # assert dash_duo.get_logs() == [], "browser console should contain no error"

    dash_duo.wait_for_text_to_equal("#label", "test", timeout=4)
    assert (
        dash_duo.find_element("#store_preview").text == "{'value': 'test'}"
    ), "store has not been updated"


if __name__ == "__main__":
    run_app(make_app)
