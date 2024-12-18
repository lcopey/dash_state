from dash import Dash, callback
from pydantic import BaseModel
from dash_state import Store, DccInput, DccLabel


def make_app() -> Dash:
    class InputState(BaseModel):
        value: str = ""

    class AppState(BaseModel):
        input_: InputState = InputState()

    input_ = DccInput(id="input", value="")
    store = Store(id="store", state_factory=AppState, storage_type="session")
    store_preview = DccLabel(id="store_preview", children="")
    debug = DccLabel(id="debug", children="")

    @store.update(input_.input)
    def on_input_change(value: str, state: AppState):
        state.input_.value = value

    @callback(store_preview.output, store.input)
    def on_store_change(state):
        return f"{state!r}"

    store.clientside_listen_on(
        "input => input.value", store.path_proxy.input_, debug.output
    )

    app = Dash()
    app.layout = [input_, store_preview, debug, store]
    return app


if __name__ == "__main__":
    app = make_app()
    app.run(debug=False)
