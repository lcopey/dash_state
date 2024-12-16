from dash import Dash, html, dcc, Input, Output, callback
from dash_state import BaseState, Store
from typing import Callable


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
    print(app.layout)
    app.run()


if __name__ == "__main__":
    run_app(make_app)
