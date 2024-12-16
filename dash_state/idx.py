"""

>>> Idx(type='store')
Idx(type='store')
>>> Idx(type='store').bind(subtype='value')
Idx(type='store', subtype='value')
>>> Idx(type='store').idx(0)
Idx(type='store', idx=0)
>>> Idx(type='store').all()
Idx(type='store', idx=<ALL>)
>>> Idx(type='store').match()
Idx(type='store', idx=<MATCH>)

"""

from typing import Any
from dash import ALL, MATCH

__all__ = ["Idx"]


class Idx(dict):
    def __init__(self, **kwargs):
        assert (
            "idx" not in kwargs
        ), f"`idx` est un mot-clé réservé et ne devrait pas se trouver dans les arguments d'appel à {self.__class__.__name__}"
        # self.kwargs = kwargs
        super().__init__(**kwargs)

    def __repr__(self):
        attrs = ", ".join(f"{key}={value!r}" for key, value in self.items())
        return f"{self.__class__.__name__}({attrs})"

    def bind(self, **kwargs):
        self.update(**kwargs)
        return self

    def idx(self, value: Any):
        return self.bind(idx=value)

    def all(self):
        return self.bind(idx=ALL)

    def match(self):
        return self.bind(idx=MATCH)
