from dash import dcc, Input, State, Output


class _IzyComponent:
    _property = None

    @property
    def input(self):
        return Input(self, self._property)

    @property
    def output(self):
        return Output(self, self._property)

    @property
    def state(self):
        return State(self, self._property)


class DccStore(_IzyComponent, dcc.Store):
    _property = "data"


class DccInput(_IzyComponent, dcc.Input):
    _property = "value"
