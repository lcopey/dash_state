from src.master_store import ReduxStore
from src.state import BaseState
from dash import Dash, html, dcc, Input, Output, State, callback
from typing import Generic, TypeVar

T = TypeVar("T", bound=type[BaseState])


class StateData(BaseState):
    class Input(BaseState):
        value: str = ""

    input: Input = Input()


Redux = ReduxStore(id="store", state_factory=StateData)
input_ = dcc.Input(id="input")
markdown = dcc.Markdown(id="markdown")

app = Dash(prevent_initial_callbacks=True)
app.layout = html.Div([Redux, input_, markdown])

Redux.store_initial(
    component_id=input_.id, component_property="value", on="value", default=""
)

"""
const obj = {
  state: {
    input: {
      value: 'ma valeur'
    }
  }
};

const path = 'state.input.value';

const getValue = (obj, path) => {
  return path.split('.').reduce((acc, part) => {
    return acc && acc[part]; // Vérifie que acc n'est pas undefined
  }, obj);
};

const value = getValue(obj, path);
console.log(value);
"""


@Redux.update(Input(input_, "value"))
def update_store(value, state: StateData):
    state.input.value = value


# @callback(
#     Output(markdown, "children"), Redux.store.as_input, prevent_initial_callback=True
# )
# def update_markdown(state):
#     print("update_markdown")
#     state = StateData.from_dict(state)
#     return state.input.value


@Redux.on_change_of(Redux.on.input.value, Output(markdown, "children"))
def update_markdown(value):
    return value


if __name__ == "__main__":
    app.run(debug=True)
