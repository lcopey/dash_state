from dash import callback as dash_callback
from dash.dependencies import DashDependency, Input, Output, State
from functools import partial
from typing import (
    Any,
    Optional,
    Callable,
    Mapping,
    Iterable,
    overload,
    TypeVar,
    Sequence,
)
from inspect import signature
from dataclasses import dataclass

__all__ = ["try_except", "filter_input", "filter_state", "filter_output", "callback"]

T = TypeVar("T", bound=DashDependency)


def try_except(func: Callable, error: type[Exception]):
    try:
        func()
    except error as e:
        print(e)


@overload
def filter_dependencies(
    dependencies: Sequence[DashDependency], dependency_type: type[T]
) -> Sequence[T]: ...


@overload
def filter_dependencies(
    dependencies: dict[str, DashDependency], dependency_type: type[T]
) -> dict[str, T]: ...


def filter_dependencies(
    dependencies: Sequence[DashDependency] | dict[str, DashDependency],
    dependency_type: type[T],
):
    if isinstance(dependencies, (tuple, list)):
        return tuple(item for item in dependencies if isinstance(item, dependency_type))
    return {k: v for k, v in dependencies.items() if isinstance(v, dependency_type)}


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


@dataclass
class NamedCallbackArgs:
    """
    >>> args = NamedCallbackArgs(
    ...     (Input('com_1', 'value'),),
    ...     (State('com_2', 'value'),),
    ...     (Output('output_1', 'value'),),
    ...     {'input': Input('com_3', 'value')},
    ...     {'state': State('com_4', 'value')},
    ...     {}
    ... )
    >>> args.to_dash_callback()
    (<Output `output_1.value`>,
    <Input `com_1.value`>,
    <Input `com_3.value`>,
    <State `com_2.value`>,
    <State `com_4.value`>)
    >>> args.to_function_call('com_1', 'com_3', 'com_2', 'com_4')
    (('com_1', 'com_2'), {'input': 'com_3', 'state': 'com_4'})
    >>> args.output_to_dash('output')
    'output'

    >>> args = NamedCallbackArgs(
    ...     (Input('com_1', 'value'),),
    ...     (State('com_2', 'value'),),
    ...     (
    ...         Output('output_1', 'value'),
    ...         Output('output_2', 'value')
    ...     ),
    ...     {'input': Input('com_3', 'value')},
    ...     {'state': State('com_4', 'value')},
    ...     {}
    ... )
    >>> args.to_dash_callback()
    (<Output `output_1.value`>,
    <Output `output_2.value`>,
    <Input `com_1.value`>,
    <Input `com_3.value`>,
    <State `com_2.value`>,
    <State `com_4.value`>)
    >>> args.output_to_dash(('result_1', 'result_2'))
    ('result_1', 'result_2')

    >>> args = NamedCallbackArgs(
    ...     (Input('com_1', 'value'),),
    ...     (State('com_2', 'value'),),
    ...     (),
    ...     {'input': Input('com_3', 'value')},
    ...     {'state': State('com_4', 'value')},
    ...     {'target': Output('result', 'value')}
    ... )
    >>> args.to_dash_callback()
    (<Output `result.value`>,
    <Input `com_1.value`>,
    <Input `com_3.value`>,
    <State `com_2.value`>,
    <State `com_4.value`>)
    >>> args.output_to_dash({'target': 'result'})
    'result'
    >>> try_except(lambda : args.output_to_dash('result'), TypeError)
    Un dictionnaire est attendu en sortie avec les clés ('target',)

    """

    pos_inputs: Sequence[Input]
    pos_states: Sequence[State]
    pos_outputs: Sequence[Output]
    kw_inputs: dict[str, Input]
    kw_states: dict[str, State]
    kw_outputs: dict[str, Output]

    def to_dash_callback(self):
        return (
            *self.pos_outputs,
            *self.kw_outputs.values(),
            *self.pos_inputs,
            *self.kw_inputs.values(),
            *self.pos_states,
            *self.kw_states.values(),
        )

    def to_function_call(self, *args: Any):
        start = 0
        sliced = []
        for items in (self.pos_inputs, self.kw_inputs, self.pos_states, self.kw_states):
            length = len(items)
            sliced.append(args[start : start + length])
            start += length
        pos_inputs, kw_inputs, pos_states, kw_states = sliced
        kw_inputs = dict(zip(self.kw_inputs.keys(), kw_inputs))
        kw_states = dict(zip(self.kw_states.keys(), kw_states))
        return (*pos_inputs, *pos_states), {**kw_inputs, **kw_states}

    def output_to_dash(self, result: Any):
        if self.pos_outputs and self.kw_outputs:
            # devrait être un tuple de tuple de dictionnaire
            pos_outputs, kw_outputs = result
            if len(self.pos_outputs) > 1:
                return *pos_outputs, *(kw_outputs[k] for k in self.kw_outputs.keys())
            else:
                return pos_outputs, *(kw_outputs[k] for k in self.kw_outputs.keys())

        if self.pos_outputs and not self.kw_outputs:
            # devrait juste être un tuple si plusieurs outputs ou la valeur simple si qu'une sortie
            # on retourne tel que
            return result

        if not self.pos_outputs and self.kw_outputs:
            # ne retourne qu'un dictionnaire dont les clés sont celles données en entrée du callback
            try:
                result = tuple(result[k] for k in self.kw_outputs.keys())
            except TypeError:
                raise CallbackError(
                    f"Un dictionnaire est attendu en sortie avec les clés {tuple(self.kw_outputs.keys())!r}"
                )

            if len(self.kw_outputs) > 1:
                return result
            else:
                return result[0]


def callback(
    *args: DashDependency,
    background: bool = False,
    interval: int = 1000,
    progress: Any = None,
    progress_default: Any = None,
    running: Any = None,
    cancel: Any = None,
    manager: Any = None,
    cache_args_to_ignore: Any = None,
    error: Optional[Callable[[Exception], Any]] = None,
    check_signature: bool = True,
    **kwargs: Output | Input | State | Any,
):
    named_callback = NamedCallbackArgs(
        pos_inputs=filter_dependencies(args, Input),
        pos_states=filter_dependencies(args, State),
        pos_outputs=filter_dependencies(args, Output),
        kw_inputs=filter_dependencies(kwargs, Input),
        kw_states=filter_dependencies(kwargs, State),
        kw_outputs=filter_dependencies(kwargs, Output),
    )
    kwargs = {
        k: v for k, v in kwargs.items() if not isinstance(v, (Input, State, Output))
    }
    kwargs["background"] = background
    kwargs["interval"] = interval
    kwargs["progress"] = progress
    kwargs["progress_default"] = progress_default
    kwargs["running"] = running
    kwargs["cancel"] = cancel
    kwargs["manager"] = manager
    kwargs["cache_args_to_ignore"] = cache_args_to_ignore
    kwargs["error"] = error

    def wrapper(func):
        if check_signature:
            pass

        @dash_callback(*named_callback.to_dash_callback(), **kwargs)
        def nested_callback(*args):
            positional_args, keyword_args = named_callback.to_function_call(*args)
            result = func(*positional_args, **keyword_args)
            return named_callback.output_to_dash(result)

        return nested_callback

    return wrapper


# def callback(
#         background: bool = False,
#         interval: int = 1000,
#         progress: Any = None,
#         progress_default: Any = None,
#         running: Any = None,
#         cancel: Any = None,
#         manager: Any = None,
#         cache_args_to_ignore: Any = None,
#         error: Optional[Callable[[Exception], Any]] = None,
#         **kwargs: Output | Input | State | Any,
# ):
#     def wrapper(func: Callable):
#         callback_kwargs = dict(
#             background=background,
#             interval=interval,
#             progress=progress,
#             progress_default=progress_default,
#             running=running,
#             cancel=cancel,
#             manager=manager,
#             cache_args_to_ignore=cache_args_to_ignore,
#             error=error,
#         )
#         targets = {}
#         inputs = {}
#         states = {}
#         for k, v in kwargs.items():
#             if not isinstance(v, DashDependency):
#                 callback_kwargs[k] = v
#             elif isinstance(v, Output):
#                 targets[k] = v
#             elif isinstance(v, Input):
#                 inputs[k] = v
#             elif isinstance(v, State):
#                 states[k] = v
#
#         _check_function_signature({**inputs, **states}, func)
#
#         @dash_callback(
#             *targets.values(), *inputs.values(), *states.values(), **callback_kwargs
#         )
#         def nested_callback(*args):
#             named_inputs = dict(zip((*inputs.keys(), *states.keys()), args))
#             results = func(**named_inputs)
#             if not isinstance(results, dict):
#                 raise CallbackError(CALLBACK_ERROR_RETURN_DICT_MSG)
#
#             results = [results[k] for k in targets.keys()]
#             if len(targets) == 1:
#                 return results[0]
#             else:
#                 return results
#
#     return wrapper
