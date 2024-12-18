from pydantic import BaseModel
from dash import Dash, html, callback
from dash_state import DccInput, DccLabel, DccStore, Store


def make_app() -> Dash:
    class AppState(BaseModel):
        value: str = ""

    input_ = DccInput(id="input0", value="")
    store = Store(id="store", state_factory=AppState, storage_type="session")
    init_store = DccStore(id="init_store", data=True, storage_type="memory")
    store_preview = DccLabel(id="store_preview")
    layout = [html.H1("Application de base"), input_, store, store_preview, init_store]

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

    app = Dash()
    app.layout = layout
    return app


if __name__ == "__main__":
    app = make_app()
    app.run(debug=False)
