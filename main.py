from src.master_store import ReduxStore
from src.state import BaseState
from dash import Dash, html, dcc, Input, Output, State, callback


class StateData(BaseState):
    class Input(BaseState):
        value: str = ""

    input: Input = Input()


# store = ReduxStore(id='store', data={}, model=StateData)
# input_ = dcc.Input(id='input')
#
# app = Dash()
# app.layout = html.Div([store, input_])
#
#
# @store.update(Input(input_, 'value'))
# def update_store(value, state: StateData):
#     return value
#
#
# if __name__ == '__main__':
#     app.run()

Redux = ReduxStore(id="store", state_factory=StateData)
input_ = dcc.Input(id="input")
markdown = dcc.Markdown(id="markdown")

app = Dash(prevent_initial_callbacks=True)
app.layout = html.Div([Redux, input_, markdown])


@Redux.update(Input(input_, "value"))
def update_store(value, state: StateData):
    state.input.value = value
    return state


@callback(Output(markdown, "children"), Redux.store.input)
def update_markdown(state):
    state = StateData.from_dict(state)
    return state.input.value


if __name__ == "__main__":
    app.run(debug=True)
