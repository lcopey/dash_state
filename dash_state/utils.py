from typing import Callable
from dash.dependencies import DashDependency, Input, Output, State
from functools import partial

__all__ = ["try_except", "filter_input", "filter_state", "filter_output"]


def try_except(func: Callable, error: type[Exception]):
    try:
        func()
    except error as e:
        print(e)


def filter_dependencies(
    *dependencies: DashDependency, dependency_type: type[DashDependency]
):
    return tuple(item for item in dependencies if isinstance(item, dependency_type))


filter_input = partial(filter_dependencies, dependency_type=Input)
filter_output = partial(filter_dependencies, dependency_type=Output)
filter_state = partial(filter_dependencies, dependency_type=State)
