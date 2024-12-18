from dash import Dash, callback, html
from pydantic import BaseModel
from dash_state import Store, DccInput, DccLabel


def make_app() -> Dash:
    class AppState(BaseModel):
        value: str = ""

    input_ = DccInput(id="input0", value="")
    store = Store(id="store", state_factory=AppState, storage_type="session")
    store_preview = DccLabel(id="store_preview")
    layout = [html.H1("Application de base"), input_, store, store_preview]

    @store.update(input_.input)
    def on_input_change(value: str, state: AppState):
        state.value = value

    # store.clientside_update(
    #     "(arg0, state) => state.value = '.'.concat(arg0);",
    #     input_0.input,
    # )

    @callback(store_preview.output, store.input)
    def update_store_preview(state: dict):
        return f"{state!r}"

    @store.init(input_.input)
    def init_input(state: AppState):
        return state.value

    app = Dash()
    app.layout = layout
    return app


if __name__ == "__main__":
    app = make_app()
    app.run(debug=False)
