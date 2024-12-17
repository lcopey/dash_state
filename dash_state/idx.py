"""
>>> from dash import Input

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
>>> Idx(type='init_store').idx((Input('input', 'value'),))
Idx(type='init_store', idx='70e842a92a447f0653f79bf869a2e550166134b03bc759d8d61b1a9a1a06958c')

"""

from dash import ALL, MATCH
from dash.dependencies import DashDependency
import orjson
from hashlib import sha256
from collections.abc import Hashable
from typing import Iterable

__all__ = ["Idx"]


def _hash_inputs(*args: DashDependency) -> Hashable:
    """
    >>> from dash import ALL, MATCH, ALLSMALLER, Input, State
    >>> _hash_inputs(Input('input', 'value'))
    '70e842a92a447f0653f79bf869a2e550166134b03bc759d8d61b1a9a1a06958c'
    >>> _hash_inputs(Input('input', 'value'), State('store', 'data'))
    'cb61779d9e17f4ea19e28ecbf9d8859626c981fb7ad8da75f158810529e109c5'
    >>> _hash_inputs(Input({'type': 'input', 'subtype': 'value'}, 'value'))
    'ec49e72fff678b546ff3c27d8de7154827b9c1eff4cd5dfcbc07fb89235f6487'
    >>> _hash_inputs(Input({'type': 'input', 'index': ALL}, 'value'))
    '70e842a92a447f0653f79bf869a2e550166134b03bc759d8d61b1a9a1a06958c'

    Args:
        *args:

    Returns:

    """
    to_hash = []
    for arg in args:
        if isinstance(arg.component_id, dict):
            id_ = "_".join(
                value
                for value in arg.component_id.values()
                if isinstance(
                    value, str
                )  # permet d'ignorer les flags ALL, MATCH, etc...
            )
        else:
            id_ = arg.component_id

        property_ = arg.component_property
        to_hash.append(".".join((id_, property_)))
    hash_ = sha256(orjson.dumps(tuple(to_hash))).hexdigest()
    return hash_


IDX_RESERVED_KW_ERROR_MSG_TEMPLATE = (
    "`idx` est un mot-clé réservé et ne "
    "doit pas être passé en argument de {class_name}"
)


class Idx(dict):
    def __init__(self, **kwargs):
        assert "idx" not in kwargs, IDX_RESERVED_KW_ERROR_MSG_TEMPLATE.format(
            self.__class__.__name__
        )
        # self.kwargs = kwargs
        super().__init__(**kwargs)

    def __repr__(self):
        attrs = ", ".join(f"{key}={value!r}" for key, value in self.items())
        return f"{self.__class__.__name__}({attrs})"

    def bind(self, **kwargs):
        self.update(**kwargs)
        return self

    def idx(self, value: str | int | Iterable[DashDependency]):
        if isinstance(value, (str, int)):
            return self.bind(idx=value)
        else:
            return self.bind(idx=_hash_inputs(*value))

    def all(self):
        return self.bind(idx=ALL)

    def match(self):
        return self.bind(idx=MATCH)

    def immutable(self):
        return tuple((k, v) for k, v in self.items())
