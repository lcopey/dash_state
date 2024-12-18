from dash import dcc, html, Input, State, Output
from dash.dependencies import DashDependency
from dash.development.base_component import Component
from dataclasses import dataclass


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


class DccLabel(html.Label):
    _property = "children"

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


class DccButton(html.Button):
    @dataclass
    class _ProxyDependency:
        dependency_type: type[DashDependency]
        component_id: str | dict | Component

        @property
        def n_clicks(self):
            return self.dependency_type(self.component_id, "n_clicks")

        @property
        def children(self):
            return self.dependency_type(self.component_id, "children")

        @property
        def disabled(self):
            return self.dependency_type(self.component_id, "disabled")

    @property
    def input(self):
        return self._ProxyDependency(Input, self)

    @property
    def output(self):
        return self._ProxyDependency(Output, self)

    @property
    def state(self):
        return self._ProxyDependency(State, self)
