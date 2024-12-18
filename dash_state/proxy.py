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
>>> proxy.value.model
<class 'str'>

>>> class NestedState(BaseModel):
...     value: str = ''
...     second_value: int = 0

>>> class AppState(BaseModel):
...     nested: NestedState = NestedState()

>>> proxy = Proxy(AppState)
>>> proxy.nested
('nested',)
>>> proxy.nested.model
<class 'dash_state.proxy.NestedState'>
>>> proxy.nested.second_value
('nested', 'second_value')
>>> try_except(lambda: proxy.invalid, KeyError)
"'invalid' is not in the fields"


>>> class Item(BaseModel):
...     text: str

>>> class AppState(BaseModel):
...     items: list[Item] = list()

>>> proxy = Proxy(AppState)
>>> proxy.items
 ('items',)
>>> proxy.items.model
list[dash_state.proxy.Item]
"""

from pydantic import BaseModel

from typing import Generic, TypeVar

T = TypeVar("T", bound=BaseModel)


class Proxy(tuple, Generic[T]):
    def __new__(cls, model: type[BaseModel] | None = None, path: tuple = tuple()):
        instance = super().__new__(cls, path)
        instance.model = model
        return instance

    @property
    def fields_info(self):
        if self.model:
            return self.model.__pydantic_fields__
        else:
            return ()

    def __getattr__(self, item):
        if item not in self.fields_info:
            raise KeyError(f"{item!r} is not in the fields")
        annotation = self.fields_info[item].annotation
        return Proxy(annotation, (*self, item))
        # try:
        #     is_terminal = ~issubclass(annotation, BaseModel)
        # except TypeError:
        #
        # if not is_terminal:
        #     return Proxy(annotation, path=(*self, item))
        # else:
        #     return Proxy(None, (*self, item))
