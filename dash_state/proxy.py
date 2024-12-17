"""
>>> from pydantic import BaseModel
>>> from dash_state.utils import try_except

>>> class AppState(BaseModel):
...     value: str
...     second_field: int

>>> proxy = Proxy(AppState)
>>> proxy.value
('value',)
>>> proxy.second_field
('second_field',)

>>> class NestedState(BaseModel):
...     value: str = ''
...     second_value: int = 0

>>> class AppState(BaseModel):
...     nested: NestedState = NestedState()

>>> proxy = Proxy(AppState)
>>> proxy.nested
('nested',)
>>> proxy.nested.second_value
('nested', 'second_value')
>>> try_except(lambda: proxy.invalid, KeyError)
"'invalid' is not in the fields"
"""

from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar("T", bound=BaseModel)


class Proxy(tuple, Generic[T]):
    def __new__(
        cls, state_factory: type[BaseModel] | None = None, path: tuple = tuple()
    ):
        instance = super().__new__(cls, path)
        instance.state_factory = state_factory
        return instance

    @property
    def fields_info(self):
        if self.state_factory:
            return self.state_factory.__pydantic_fields__
        else:
            return ()

    def __getattr__(self, item):
        if item not in self.fields_info:
            raise KeyError(f"{item!r} is not in the fields")
        annotation = self.fields_info[item].annotation
        if issubclass(annotation, BaseModel):
            return Proxy(annotation, path=(*self, item))
        else:
            return Proxy(None, (*self, item))
