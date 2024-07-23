"""
This module contains the Logger class which is used to log messages to the console.

Methods:
    - info: Log an info message to the console.
    - error: Log an error message to the console.
"""

from tmll.common.utils.colors import ConsoleColors

class Logger:
    def __init__(self, name: str, verbose: bool = True) -> None:
        self.name = name
        self.verbose = verbose

    def info(self, msg: str) -> None:
        if self.verbose:
            print(f"{ConsoleColors.BOLD}{ConsoleColors.OKBLUE}[INFO]{ConsoleColors.ENDC} {ConsoleColors.OKGREEN}{self.name}:{ConsoleColors.ENDC} {msg}")

    def error(self, msg: str) -> None:
        if self.verbose:
            print(f"{ConsoleColors.BOLD}{ConsoleColors.FAIL}[ERROR]{ConsoleColors.ENDC} {ConsoleColors.OKGREEN}{self.name}:{ConsoleColors.ENDC} {msg}")

    def warning(self, msg: str) -> None:
        if self.verbose:
            print(f"{ConsoleColors.BOLD}{ConsoleColors.WARNING}[WARNING]{ConsoleColors.ENDC} {ConsoleColors.OKGREEN}{self.name}:{ConsoleColors.ENDC} {msg}")
