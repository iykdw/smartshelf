import sqlite3
from contextlib import closing
from typing import List
from typing import Optional

class DB:
    def __init__(self, persist, file, logging):
        self.persist = persist
        self.logging = logging
        self.file = file

    def log_trace(self, statement):
        self.logging.info(statement)

    def _db_execute(self, command: str, args: tuple, com_type: int) -> Optional[List]:
        with closing(sqlite3.connect(f"{self.persist}/{self.file}")) as connection:
            connection.set_trace_callback(self.log_trace)
            with closing(connection.cursor()) as c:
                if com_type == 2:
                    return c.execute(command, args).fetchall()
                elif com_type == 1:
                    return c.execute(command, args).fetchone()
                else:
                    c.execute(command, args)

                connection.commit()


    def execute(self, command: str, args: tuple = ()) -> None:
        self.logging.debug(command)
        self.logging.debug(args)
        self._db_execute(command, args, 0)


    def fetchone(self, command: str, args: tuple = ()) -> List | None:
        result = self._db_execute(command, args, 1)
        return result

    def fetchall(self, command: str, args: tuple = ()) -> List | None:
        return self._db_execute(command, args, 2)
