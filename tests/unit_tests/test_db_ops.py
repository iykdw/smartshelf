import logging
import os
import subprocess
from pathlib import Path

import pytest

import smartshelf_logger
from src.db_ops import DB as db_ops_DB

cwd = os.getcwd()

test_dir = Path(cwd) / "tests"
test_db = Path(test_dir) / "test.db"
test_logfile = Path(test_dir) / "test.log"
schemafile = Path(cwd) / "table_schema.sql"

log_files = [test_logfile.resolve()]
logger = smartshelf_logger.get_logger(log_files, [], logging.DEBUG, __file__)
logger.info("Hello, world!")


@pytest.fixture
def DB():
    import json

    from src.main import _has_config, format_book_for_db_insertion, get_data_from_gb, get_rooms, suggest_position

    DB_handler = db_ops_DB(test_db, logger)

    if _has_config(DB_handler):
        logger.info("DB exists and is set up correctly!")
        return DB_handler

    logger.info("Creating DB now...")
    command = [f'uv run sqlite3 {test_db.resolve()} ".read {schemafile.resolve()}"']
    logger.info(command)
    subprocess.run(command)
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


def test_idk(DB):
    logger.info(os.getcwd())
