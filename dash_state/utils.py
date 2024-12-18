from dash import callback as dash_callback
from dash.dependencies import DashDependency, Input, Output, State
from functools import partial
from typing import Any, Optional, Callable, Mapping
from inspect import signature

__all__ = ["try_except", "filter_input", "filter_state", "filter_output", "callback"]


def try_except(func: Callable, error: type[Exception]):
    try:
        func()
    except error as e:
        print(e)


def filter_dependencies(
    *dependencies: DashDependency, dependency_type: type[DashDependency]
):
    return tuple(item for item in dependencies if isinstance(item, dependency_type))


class CallbackError(TypeError): ...


filter_input = partial(filter_dependencies, dependency_type=Input)
filter_output = partial(filter_dependencies, dependency_type=Output)
filter_state = partial(filter_dependencies, dependency_type=State)

CALLBACK_ERROR_RETURN_DICT_MSG = """
Le résultat de `callback` doit être un dictionnaire dont les clés correspondent au 
nom des outputs passés en argument : 

@callback(target=Output(...), value=Input(...))
def foo(value):
    ...
    return {'target': ...}

"""

CALLBACK_ERROR_WRONG_SIGNATURE_MSG = """
Le callback qui décore la fonction {function} attendait les paramètres suivants : 
{expected} mais a reçu les suivants {actual}
"""


def _check_function_signature(expected: Mapping, function: Callable):
    actual = list(signature(function).parameters.keys())
    expected = list(expected.keys())

    diff = set(expected).difference(set(actual))
    if diff:
        raise CallbackError(
            CALLBACK_ERROR_WRONG_SIGNATURE_MSG.format(
                function=function, expected=expected, actual=actual
            )
        )

    diff = set(actual).difference(set(expected))
    if diff:
        raise CallbackError(
            CALLBACK_ERROR_WRONG_SIGNATURE_MSG.format(
                function=function, expected=expected, actual=actual
            )
        )


def callback(
    background: bool = False,
    interval: int = 1000,
    progress: Any = None,
    progress_default: Any = None,
    running: Any = None,
    cancel: Any = None,
    manager: Any = None,
    cache_args_to_ignore: Any = None,
    error: Optional[Callable[[Exception], Any]] = None,
    **kwargs: Output | Input | State | Any,
):
    def wrapper(func: Callable):
        callback_kwargs = dict(
            background=background,
            interval=interval,
            progress=progress,
            progress_default=progress_default,
            running=running,
            cancel=cancel,
            manager=manager,
            cache_args_to_ignore=cache_args_to_ignore,
            error=error,
        )
        targets = {}
        inputs = {}
        states = {}
        for k, v in kwargs.items():
            if not isinstance(v, DashDependency):
                callback_kwargs[k] = v
            elif isinstance(v, Output):
                targets[k] = v
            elif isinstance(v, Input):
                inputs[k] = v
            elif isinstance(v, State):
                states[k] = v

        _check_function_signature({**inputs, **states}, func)

        @dash_callback(
            *targets.values(), *inputs.values(), *states.values(), **callback_kwargs
        )
        def nested_callback(*args):
            named_inputs = dict(zip((*inputs.keys(), *states.keys()), args))
            results = func(**named_inputs)
            if not isinstance(results, dict):
                raise CallbackError(CALLBACK_ERROR_RETURN_DICT_MSG)

            results = [results[k] for k in targets.keys()]
            if len(targets) == 1:
                return results[0]
            else:
                return results

    return wrapper
