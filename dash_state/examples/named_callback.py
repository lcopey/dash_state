from dash import Dash
from dash_state.utils import callback
from dash_state.fast_dependencies import DccLabel, DccInput, DccButton


def make_app():
    input_ = DccInput(id="input", value="")
    label = DccLabel(id="label", children="")
    button = DccButton(id="button", children="click")

    app = Dash()
    app.layout = [input_, label, button]

    @callback(
        target=label.output,
        input=input_.state,
        n_click=button.input.n_clicks,
        prevent_initial_call=True,
    )
    def on_click(
        n_clicks: int,
        input: str,
    ):
        return {"target": f"{n_clicks} {input}"}

    return app


if __name__ == "__main__":
    app = make_app()
    app.run(debug=False)
