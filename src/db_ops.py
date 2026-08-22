import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


class DB:
    def __init__(self, db_file: Path, logging):
        self.logging = logging
        self.file = db_file
        self.logging.debug(self.fetchall("""SELECT name FROM sqlite_master WHERE type='table';"""))

    def log_trace(self, statement: str):
        self.logging.debug(statement)

    def _db_execute(self, command: str, args: tuple[str, ...], com_type: int) -> list[None | list[str]]:
        with closing(sqlite3.connect(self.file.resolve())) as connection:
            connection.set_trace_callback(self.log_trace)
            with closing(connection.cursor()) as c:
                if com_type == 2:
                    return c.execute(command, args).fetchall()
                if com_type == 1:
                    return c.execute(command, args).fetchone()
                _ = c.execute(command, args)
                connection.commit()
                return []

    def execute(self, command: str, args: tuple[Any, ...] | None = tuple()) -> None:
        if not args:
            args = tuple()
        self.logging.debug(command)
        self.logging.debug(args)
        self._db_execute(command, args, 0)

    def fetchone(self, command: str, args: tuple[Any, ...] | None = None) -> list:
        if not args:
            args = tuple()
        return self._db_execute(command, args, 1)

    def fetchall(self, command: str, args: tuple[str, ...] | None = None) -> list:
        if not args:
            args = tuple()
        return self._db_execute(command, args, 2)
