"""
# Définition de l'état de l'application

L'état de l'application est définie sur base de pydantic
>>> from pydantic import BaseModel

>>> class AppState(BaseModel):
...     value: str = ''

>>> AppState()
AppState(value='')

Il est possible de définir des états imbriqués

>>> class InnerState(BaseModel):
...     nested_value: str = ''

>>> class AppState(BaseModel):
...     value: InnerState = InnerState()

>>> AppState()
AppState(value=InnerState(nested_value=''))

Les valeurs passées par défault peuvent être des valeurs mutables également

>>> class InnerState(BaseModel):
...     nested_sequence: list = []

>>> class AppState(BaseModel):
...     value: InnerState = InnerState()

>>> AppState()
AppState(value=InnerState(nested_sequence=[]))
"""
