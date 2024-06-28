"""
This module contains the Logger class which is used to log messages to the console.

Methods:
    - info: Log an info message to the console.
    - error: Log an error message to the console.
"""

class Logger:
    def __init__(self, name: str, verbose: bool = True) -> None:
        self.name = name
        self.verbose = verbose

    def info(self, msg: str) -> None:
        if self.verbose:
            print(f"[INFO] {self.name}: {msg}")

    def error(self, msg: str) -> None:
        if self.verbose:
            print(f"[ERROR] {self.name}: {msg}")
