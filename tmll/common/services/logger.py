"""
This module contains the Logger class which is used to log messages to the console.

Methods:
    - info: Log an info message to the console.
    - error: Log an error message to the console.
"""

import builtins
from functools import wraps

from tmll.common.utils.colors import ConsoleColors

"""
In this code block, we want to disable the print function globally, except for the Logger class.
"""
_original_print = builtins.print

def enable_printing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        builtins.print = _original_print
        try:
            return func(*args, **kwargs)
        finally:
            builtins.print = lambda *args, **kwargs: None
    return wrapper
builtins.print = lambda *args, **kwargs: None

class Logger:
    def __init__(self, name: str, verbose: bool = True) -> None:
        self.name = name
        self.verbose = verbose

    def _format_message(self, level: str, msg: str) -> str:
        return (f"{ConsoleColors.BOLD}{level}{ConsoleColors.ENDC} "
                f"{ConsoleColors.OKGREEN}{self.name}:{ConsoleColors.ENDC} {msg}")

    def info(self, msg: str) -> None:
        self._log_message(f"{ConsoleColors.OKBLUE}[INFO]{ConsoleColors.ENDC}", msg)

    def error(self, msg: str) -> None:
        self._log_message(f"{ConsoleColors.FAIL}[ERROR]{ConsoleColors.ENDC}", msg)

    def warning(self, msg: str) -> None:
        self._log_message(f"{ConsoleColors.WARNING}[WARNING]{ConsoleColors.ENDC}", msg)

    @enable_printing
    def _log_message(self, level: str, msg: str) -> None:
        if self.verbose:
            print(self._format_message(level, msg))