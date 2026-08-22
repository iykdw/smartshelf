import logging
import os
from pathlib import Path

import pytest

import smartshelf_logger
from rooms_to_db import rooms_to_db
from src.db_ops import DB as db_ops_DB

cwd = os.getcwd()

test_dir = Path(cwd) / "tests"
db_file = Path(test_dir) / "test.db"
test_logfile = Path(test_dir) / "test.log"
schemafile = Path(cwd) / "table_schema.sql"
rooms_file = Path(test_dir) / "rooms.json"

log_files = [test_logfile.resolve()]
logger = smartshelf_logger.get_logger(log_files, [], logging.DEBUG, __file__)
logger.info("Hello, world!")


@pytest.fixture
def DB():
    import json

    from src.main import _has_config, format_book_for_db_insertion, get_data_from_gb, get_rooms, suggest_position

    DB_handler = db_ops_DB(db_file.resolve(), logger)

    if _has_config(DB_handler):
        logger.info("DB exists and is set up correctly!")
        return DB_handler

    logger.info("Creating DB now...")
    os.system(f'uv run sqlite3 {db_file.resolve()} ".read {schemafile.resolve()}"')
    rooms_to_db(DB, rooms_file.resolve())
    logger.info(" done")
    with open("tests/isbns.json") as f:
        isbns = json.loads(f.read())
        rooms = get_rooms()
        for isbn in isbns:
            book = DB_handler.fetchone("""SELECT COUNT(*) FROM books WHERE isbn = ?""", (isbn,))
            room = next(iter(rooms.keys()))
            if book[0] == 0:
                book = get_data_from_gb(isbn)
                book, _ = suggest_position(book, rooms[room].shelves, DB_handler)
                book.room = room
                print(book)
                DB_handler.execute(
                    """INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    format_book_for_db_insertion(book),
                )

    return DB_handler
