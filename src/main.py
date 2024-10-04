# from redux_store import ReduxStore
# from dash import Dash, html, dcc, Input, Output, State
# from pydantic import BaseModel
#
# # class StateData(BaseModel):
# #     class Input(BaseModel):
# #         value: str = ''
# #
# #     input: Input = Input()
# #
# #
# # store = ReduxStore(id='store', data={}, model=StateData)
# # input_ = dcc.Input(id='input')
# #
# # app = Dash()
# # app.layout = html.Div([store, input_])
# #
# #
# # @store.update(Input(input_, 'value'))
# # def update_store(value, state: StateData):
# #     return value
# #
# #
# # if __name__ == '__main__':
# #     app.run()
#
# DATA_MODEL = {"input": {"value": ""}}
#
# store = ReduxStore(id="store", data=DATA_MODEL)
# input_ = dcc.Input(id="input")
#
# app = Dash()
# app.layout = html.Div([store, input_])
#
#
# @store.update(Input(input_, "value"))
# def update_store(value, state):
#     state["input"]["value"] = value
#     return state
#
#
# if __name__ == "__main__":
#     app.run()
