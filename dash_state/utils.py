from typing import Callable


def try_except(func: Callable, error: type[Exception]):
    try:
        func()
    except error as e:
        print(e)
