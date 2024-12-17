from dash import dcc, Input, State, Output
from dash.dependencies import DashDependency


# class _IzyComponent:
#     _property = None
#
#     @property
#     def input(self):
#         return Input(self, self._property)
#
#     @property
#     def output(self):
#         return Output(self, self._property)
#
#     @property
#     def state(self):
#         return State(self, self._property)


class DccStore(dcc.Store):
    _property = "data"

    @property
    def input(self):
        return Input(self, self._property)

    @property
    def output(self):
        return Output(self, self._property)

    @property
    def state(self):
        return State(self, self._property)

    @property
    def component(self):
        return DashDependency(self, self._property)


class DccInput(dcc.Input):
    _property = "value"

    @property
    def input(self):
        return Input(self, self._property)

    @property
    def output(self):
        return Output(self, self._property)

    @property
    def state(self):
        return State(self, self._property)

    @property
    def component(self):
        return DashDependency(self, self._property)
