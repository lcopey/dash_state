# Features

- Objet State
  - Servira de centralisation de l'état de l'application dans une définition et instanciation centrale
    - Store et objet python miroir
  - Servira d'état interne des différents widgets
  - Est réinstantié au démarrage de l'appli
  - Comprends plusieurs niveaux :
```python
class BaseState:
  ...


class InputState(BaseState):
  value: str


class ApplicationState(BaseState):
  input_state: InputState
  value: str
```
  - La modification d'un attribut direct de BaseState déclenche un évènement servant à mettre à jour un comopsant

```python
class BaseState:
  ...


class InputState(BaseState):
  value: str


class ApplicationState(BaseState):
  input_state: InputState
  value: str

state = ApplicationState()
@on_change(state.input_state.value, Output(component_id))
def on_input_change(state: InputState):
    ...
```
  - La modification d'un composant sert également à mettre à jour l'état de l'application
