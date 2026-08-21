import copy
import json
import logging
import os
import pprint
import random
import re
import sqlite3
import sys
import time
from datetime import datetime
from typing import Annotated
from urllib.error import HTTPError
from urllib.request import urlopen
from uuid import uuid4

from fastapi import FastAPI, Form, Request, WebSocket, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

import db_ops
from models import Book, Room, Shelf
from natlangposition import nat_lang_position
from rooms_to_db import rooms_to_db


class CouldNotShelveError(Exception):
    pass


class DatabaseNotFoundError(Exception):
    pass


class UnconfiguredError(Exception):
    pass


class Secrets(BaseSettings):
    GOOGLE_BOOKS_API_KEY: SecretStr

    model_config = SettingsConfigDict(env_file=".env")


def has_config(DB):
    try:
        DB.fetchone("""SELECT COUNT(*) FROM books""")
        DB.fetchone("""SELECT COUNT(*) FROM transactions""")
        DB.fetchone("""SELECT COUNT(*) FROM users""")
        DB.fetchone("""SELECT COUNT(*) FROM rooms""")
        DB.fetchone("""SELECT COUNT(*) FROM shelves""")
    except sqlite3.OperationalError:
        return False

    return True


def get_data_from_api(url: str):
    logger.info(f"Requesting data from URL {url}")
    try:
        response = urlopen(url)
        logger.info("Response received!")
        return response
    except HTTPError as e:
        logger.info(f"Error while requesting data:\n    Code: {e.code}\n    Reason: {e.reason}\n    Headers:\n{'\n    '.join(e.headers)}")
        return e
        return {"code": e.code, "reason": e.reason, "headers": e.headers.split()}


def get_data_from_gb(isbn) -> Book:
    if len(isbn) not in [10, 13]:
        logger.info(f"Submitted string {isbn} isn't a valid ISBN.")
        # TODO: raise an error properly

    logger.info(f"Searching Google Books for ISBN {isbn}.")
    isbn = "".join([char for char in isbn if char.isdigit()])

    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}&projection=full&key={secrets.GOOGLE_BOOKS_API_KEY.get_secret_value()}"
    logger.info("Requesting data from Google Books")
    response = get_data_from_api(url)
    if isinstance(response, HTTPError):
        return response

    book_data = json.load(response)

    if book_data["totalItems"] == 0:
        return Book(
            uuid=str(uuid4()),
            isbn=isbn,
            title="",
            subtitle="",
            author="",
            pages=0,
            width=0,
            room="",
            shelf="",
            position=-1,
            withdrawn="",
        )

    book_id = book_data["items"][0]["id"]

    url = f"https://www.googleapis.com/books/v1/volumes/{book_id}?key={secrets.GOOGLE_BOOKS_API_KEY.get_secret_value()}"
    logger.info(f"Book has ID {book_id}")
    response = get_data_from_api(url)

    book_data = json.load(response)["volumeInfo"]
    book = Book(
        uuid=str(uuid4()),
        isbn=isbn,
        title=book_data["title"],
        subtitle=book_data["subtitle"] if "subtitle" in book_data else "",
        author=", ".join(book_data["authors"]) if "authors" in book_data else "",
        pages=book_data["printedPageCount"] if "printedPageCount" in book_data else 0,
        width=int((int(book_data["printedPageCount"]) if "printedPageCount" in book_data else 0) * mm_per_page),
        room="",
        shelf="",
        position=-1,
        withdrawn="",
    )

    logger.info(f"Book {book.title} has been assigned uuid {book.uuid}.")
    logger.info("Data read, returning now.")

    return book


def suggest_position(book: Book, room_shelves: dict[str, Shelf], DB: db_ops.DB, requested_shelf: str = "") -> tuple[Book, str]:
    shelves = []

    if requested_shelf != "":
        req_shelf = room_shelves[requested_shelf]
        shelves.append(req_shelf)
        del room_shelves[requested_shelf]
        shelves += list(room_shelves.values())
    else:
        shelves = list(room_shelves.values())

    for shelf in shelves:
        books_on_shelf: list[tuple[str, int, int]] = DB.fetchall(
            """SELECT title, width, position FROM books WHERE shelf = ? ORDER BY position""",
            (shelf.uuid,),
        )

        neighbour = "shelf edge"

        if books_on_shelf == []:  # i.e. if shelf is completely empty
            logger.info(f"    Book {book.title} can be shelved on {shelf.name} ({shelf.uuid}) at {int(book.width / 2)} (shelf is empty)")
            book.shelf = shelf.uuid
            book.position = int(book.width / 2)
            book.natlangpos = "at the far left"
            book.shelf_name = shelf.name
            return (
                book,
                neighbour,
            )
        # if shelf is not empty
        empty: list[int] = list(range(shelf.width))
        logging.debug(shelf)
        logging.debug(f"Attempting to shelve {book.title} on shelf {shelf.uuid}")

        for shelved_book in books_on_shelf:
            hw = int(shelved_book[1] / 2)  # halved width
            start = int(shelved_book[2] - hw)
            end = int(shelved_book[2] + hw)
            logging.debug(f"    {shelved_book[0]} in position {start}-{end}")

            if start in empty:
                curr = start  # remove the millimeters occupied by this book
            else:
                curr = start + 1

            while curr <= end:
                try:
                    empty.remove(curr)
                except ValueError:
                    logging.debug("that fucking empty.remove(curr) bug, bestie, fix it")

                curr += 1

        # gaps will be stored as [start_index, end_index]
        gaps = []

        for j in range(len(empty) - 1):
            # Iterate over all the millimetres not currently occupied by a book

            if len(gaps) == 0 and empty[0] == 0:
                # If there are no gaps known and the first gap is at the edge of the shelf, start a new gap at -1 because... offset reasons
                gaps.append([-1])
            elif len(gaps) == 0:
                # If there are no gaps known, but the first gap starts inset from the edge of the shelf
                gaps.append([empty[0]])

            if len(gaps[-1]) == 2:
                # If the current gap is complete, start a new gap
                gaps.append([empty[j]])

            if empty[j] != empty[j + 1] - 1:
                # If the next empty millimetre is not the next millimetre, finish this gap.
                gaps[-1].append(empty[j])

        if len(gaps) == 0:
            logger.info(f"    Shelf {shelf.name} ({shelf.uuid}) completely full; proceeding to next shelf")

            continue

        gaps[-1].append(empty[-1])
        logging.debug(f"    Gaps found - {gaps}")

        gaps = sorted(gaps, key=lambda x: x[1] - x[0], reverse=True)
        # Find the largest gap
        largest = gaps[0][1] - gaps[0][0]
        logging.debug(
            f"    The largest gap on shelf {shelf.name} ({shelf.uuid}) is {largest}mm. Book is {book.width}mm wide and would {'not ' if book.width > largest else ''}fit."
        )

        if (gaps[0][1] - gaps[0][0]) < book.width:
            # if the biggest gap is too small, proceed to next shelf
            logger.info(f"    Book {book.title} cannot be shelved on shelf {shelf.name} ({shelf.uuid})")
            continue

        suggested = gaps[0][0] + int(book.width / 2)
        book.shelf = shelf.uuid
        book.shelf_name = shelf.name
        book.position = suggested
        if gaps[0][0] == -1:
            book.natlangpos = "at the far left"
        else:
            book.natlangpos = nat_lang_position(book.position, shelf.width)

        logger.info(f"    Book {book.title} can be shelved on {shelf.name} ({shelf.uuid}) at {suggested}")

        # When would gaps[0][1] ever be undefined?
        if gaps[0][0] == -1 and gaps[0][1]:
            return (
                book,
                neighbour,
            )

        neighbour = books_on_shelf[0][0]

        for shelved in books_on_shelf:
            if shelved[2] > book.position:  # Keep going until we overshoot, then return the last
                return (
                    book,
                    neighbour,
                )
            neighbour = shelved[0]

        return (  # The previous will fail if this book is the last book on the shelf!
            book,
            neighbour,
        )

    raise CouldNotShelveError


def get_rooms() -> dict[str, Book]:
    room_data = DB.fetchall("""SELECT * FROM rooms""")
    rooms = {}
    for r in room_data:
        room = Room(uuid=r[0], name=r[1], shelves={})
        shelves = get_shelves(r[0])
        for shelf in shelves:
            room.shelves[shelf.uuid] = shelf
        rooms[r[0]] = room

    return rooms


def get_shelves(room: str = "") -> list[Shelf]:
    shelves = []
    if room == "":
        shelf_data = DB.fetchall("""SELECT * FROM shelves""")

    else:
        shelf_data = DB.fetchall("""SELECT * FROM shelves WHERE room = ?""", (room,))

    for s in shelf_data:
        shelves.append(Shelf(uuid=s[0], name=s[1], width=s[2], room=s[3]))

    return shelves


def format_book_for_db_insertion(book: Book) -> tuple[str | int]:
    return_data = (
        book.uuid,
        book.isbn,
        book.title,
        book.subtitle,
        book.author,
        book.pages,
        book.room,
        book.shelf,  # Shelf
        book.position,  # Position
        book.width,
        book.withdrawn,  # Status
    )

    return return_data


def format_db_record_as_book(record: tuple[str | int]) -> Book:
    return Book(
        uuid=record[0],
        isbn=record[1],
        title=record[2],
        subtitle=record[3],
        author=record[4],
        pages=record[5],
        room=record[6],
        shelf=record[7],
        position=record[8],
        width=record[9],
        withdrawn=record[10],
    )


def _build_db(DB):
    os.system(f'uv run sqlite3 {persist_dir}/{db_file} ".read table_schema.sql"')
    if len(get_rooms().keys()) == 0:
        rooms_to_db(DB, f"{persist_dir}/rooms.json")
    return
    # Only if we want to populate the db
    with open(f"{persist_dir}/isbns.json") as f:
        isbns = json.loads(f.read())
        for isbn in isbns:
            book = DB.fetchone("""SELECT COUNT(*) FROM books WHERE isbn = ?""", (isbn,))
            rooms = get_rooms()
            room = list(rooms.keys())[0]
            if book[0] == 0:
                book = get_data_from_gb(isbn)
                book, _ = suggest_position(book, rooms[room].shelves, DB)
                book.room = room
                print(book)
                DB.execute(
                    """INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    format_book_for_db_insertion(book),
                )


default_user = "yasha"
persist_dir = "storage"
db_file = "books.db"
mm_per_page = 0.0696729243
secrets = Secrets()

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f"{persist_dir}/debug.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

REDACT_REGEXES = [r"(key=[ ]?)([^&]*)"]


class LoggingFilter(logging.Filter):
    def __init__(self, patterns):
        super().__init__()
        self.patterns = patterns

    def filter(self, record: logging.LogRecord) -> bool:
        for pattern in self.patterns:
            record.msg = re.sub(pattern, "<REDACTED>", record.msg)
        return True


logger = logging.getLogger(__name__)
logger.addFilter(LoggingFilter(REDACT_REGEXES))
logger.info("Hello, world!")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DB = db_ops.DB(persist_dir, db_file, logging)

_build_db(DB)

logger.info(f"Checking db at {persist_dir}/{db_file}...")
if has_config(DB):
    logger.info("DB check complete.")
else:
    raise DatabaseNotFoundError


@app.get("/", response_class=HTMLResponse)
def get_library(request: Request):
    if request["type"] == "https":
        ws_address = f"wss://{str(request.url).split('/')[2]}/search"
    else:
        ws_address = f"ws://{str(request.url).split('/')[2]}/search"

    if not has_config(DB):
        raise UnconfiguredError

    books_raw = DB.fetchall("""SELECT * FROM books""")[::-1]

    books = []

    for book_data in books_raw:
        books.append(format_db_record_as_book(book_data))

    return templates.TemplateResponse(
        request=request,
        name="library.html",
        context={"books": books, "ws_address": ws_address},
    )


@app.get("/populate/{isbn}")
def add_book(isbn: str):
    book_data = get_data_from_gb(isbn)
    backoff = 1
    while isinstance(book_data, HTTPError):
        if book_data.code == 503:
            this_backoff = backoff + random.randint(1, 10) / 10
            logger.info(f"Server 503'd, sleeping for {this_backoff}s.")
            this_backoff = backoff + random.randint(1, 10) / 10
            if backoff < 4:
                backoff = backoff * 2
            book_data = get_data_from_gb(isbn)
        else:
            return None

    DB.execute(
        """INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        format_book_for_db_insertion(book_data),
    )

    uuid = book_data.uuid
    return RedirectResponse(url=f"/book/{uuid}", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/book/{uuid}", response_class=HTMLResponse)
def hit_endpoint(request: Request, uuid: str):
    book_data = DB.fetchone("""SELECT * FROM books WHERE uuid = ?""", (uuid,))

    if not book_data:  # If this book is being added
        return RedirectResponse(url=f"/add/{uuid}", status_code=status.HTTP_302_FOUND)

    book = format_db_record_as_book(book_data)
    return templates.TemplateResponse(request=request, name="book.html", context={"book": book})


@app.get("/edit/{uuid}", response_class=HTMLResponse)
def edit_book(request: Request, uuid: str):
    book_data = format_db_record_as_book(DB.fetchone("""SELECT * FROM books WHERE uuid = ? """, (uuid,)))

    return templates.TemplateResponse(
        request=request,
        name="validate.html",
        context={
            "book": book_data,
            "rooms": [get_rooms()[room].name for room in get_rooms().keys()],
            "mm_per_page": mm_per_page,
        },
    )


@app.post("/update")
async def update_book(book: Annotated[Book, Form()]):
    print(book)
    if book.time != "0":
        time = round(float(book.time))

        if book.withdrawn == "withdrawn":
            DB.execute(
                """INSERT INTO transactions VALUES (?, ?, ?, ?, ?)""",
                (book.isbn, "withdrawn", 0, time, book.user),
            )

            book.withdrawn = f"{datetime.fromtimestamp(time).strftime('%Y-%m-%d')} by {book.user}"

        elif book.withdrawn == "shelved":
            DB.execute(
                """INSERT INTO transactions VALUES (?, ?, ?, ?, ?)""",
                (book.isbn, "shelved", 1, time, book.user),
            )

            book.withdrawn = ""

    DB.execute(
        """INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        format_book_for_db_insertion(book),
    )

    if book.shelf == -2:
        return RedirectResponse(url=f"/shelve/{book.isbn}", status_code=status.HTTP_303_SEE_OTHER)

    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/shelve/{book_uuid}", response_class=HTMLResponse)
async def shelve(book_uuid: str, request: Request):
    if request["type"] == "https":
        ws_address = f"wss://{str(request.url).split('/')[2]}/shelve"
    else:
        ws_address = f"ws://{str(request.url).split('/')[2]}/shelve"

    book_data = format_db_record_as_book(DB.fetchone("""SELECT * FROM books WHERE uuid = ? """, (book_uuid,)))

    rooms = get_rooms()
    book_data.withdrawn = ""

    position_found = False

    possible_rooms = []

    while not position_found:
        room_uuid = next(iter(list(rooms.keys())))
        room = rooms[room_uuid]

        logger.info(room.name)

        viable_shelves = {}
        for shelf in list(room.shelves.values()):
            try:
                _book_data, _neighbour = suggest_position(copy.copy(book_data), {shelf.uuid: room.shelves[shelf.uuid]}, DB)
            except CouldNotShelveError:
                continue

            viable_shelves[shelf.uuid] = shelf
            if not position_found:
                position_found = True
                book_data = _book_data
                neighbour = _neighbour
            if room.uuid not in possible_rooms:
                possible_rooms.append(room.uuid)

        logger.info(
            f"    {book_data.title} can be shelved on {len(viable_shelves.keys())} out of a possible {len(room.shelves)} in {room.name}."
        )

    shelves_list = []
    for shelf in room.shelves.values():
        shelves_list.append({"name": shelf.name, "uuid": shelf.uuid, "disabled": False if shelf.uuid in viable_shelves else True})

    rooms_list = []
    for room in rooms.values():
        rooms_list.append({"name": room.name, "uuid": room.uuid, "disabled": False if room.uuid in possible_rooms else True})

    context_dict = {
        "book": book_data,
        "neighbour": neighbour,
        "ws_address": ws_address,
        "rooms": rooms_list,
        "shelves": shelves_list,
        "time": time.time(),
    }

    logger.info(f"Preloading /shelve with values {pprint.pformat(context_dict)}")

    return templates.TemplateResponse(request=request, name="shelve.html", context=context_dict)


@app.get("/withdraw/{uuid}", response_class=HTMLResponse)
def withdraw(uuid: str, request: Request):
    user = request.headers.get("Remote-Name")
    if user is None:
        user = default_user
    book_data = format_db_record_as_book(DB.fetchone("""SELECT * FROM books WHERE uuid = ? """, (uuid,)))

    DB.execute("INSERT OR REPLACE INTO USERS VALUES (?)", (user,))
    users = [user[0] for user in DB.fetchall("""SELECT name FROM users""")]

    book_data.shelf = ""
    book_data.position = -1

    return templates.TemplateResponse(
        request=request,
        name="withdraw.html",
        context={"book": book_data, "users": users, "time": time.time(), "user_name": user},
    )


@app.get("/locate/{uuid}", response_class=HTMLResponse)
def locate(uuid: str, request: Request):
    book = format_db_record_as_book(DB.fetchone("""SELECT * FROM books WHERE uuid = ? """, (uuid,)))
    logger.info(str(book))

    if book.natlangpos == "":
        shelf = book.shelf
        room = get_rooms()[book.room]
        shelf = room.shelves[book.shelf]
        book.natlangpos = nat_lang_position(book.position, shelf.width)

    if book.shelf_name == "":
        room = get_rooms()[book.room]
        shelf = room.shelves[book.shelf]
        book.shelf_name = shelf.name

    return templates.TemplateResponse(
        request=request,
        name="locate.html",
        context={"book": book},
    )


# TODO: client-side nag if room name already exists
@app.get("/addroom", response_class=HTMLResponse)
def add_room(request: Request):
    rooms = get_rooms()
    return templates.TemplateResponse(
        request=request,
        name="add_location.html",
        context={
            "loctype": "room",
            "rooms": rooms,
        },
    )


@app.post("/roomadd")
def _add_room(room: Annotated[Room, Form()]):
    room.uuid = str(uuid4())

    logger.info(f"Adding room {room.name} with uuid {room.uuid}.")
    DB.execute(
        """INSERT INTO rooms VALUES (?, ?)""",
        (
            room.uuid,
            room.name,
        ),
    )

    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


# TODO: client-side nag if shelf name already exists in room
@app.get("/addshelf", response_class=HTMLResponse)
def add_shelf(request: Request):
    rooms = get_rooms()
    shelves = get_shelves()
    return templates.TemplateResponse(
        request=request,
        name="add_location.html",
        context={
            "loctype": "shelf",
            "rooms": [room.dict() for room in rooms],
            "shelves": [shelf.dict() for shelf in shelves],
        },
    )


@app.post("/shelfadd")
def _add_shelf(shelf: Annotated[Shelf, Form()]):
    shelf.uuid = str(uuid4())
    room = DB.fetchone("""SELECT name FROM rooms WHERE id LIKE ?""", (shelf.room,))

    logger.info(f"Adding shelf {shelf.name} with uuid {shelf.uuid} in room {room}.")
    DB.execute(
        """INSERT INTO shelves VALUES (?, ?, ?, ?)""",
        (
            shelf.uuid,
            shelf.name,
            int(shelf.width),
            shelf.room,
        ),
    )

    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@app.websocket("/search")
async def search_websocket(websocket: WebSocket):
    await websocket.accept()

    while True:
        query = await websocket.receive_text()
        wild_query = f"%{query}%"
        results = DB.fetchall(
            """SELECT uuid, title, author, withdrawn FROM books WHERE title LIKE ? OR subtitle LIKE ? OR author LIKE ?""",
            (wild_query, wild_query, wild_query),
        )
        await websocket.send_json(results)


@app.websocket("/shelve")
async def shelve_websocket(websocket: WebSocket):
    await websocket.accept()

    while True:
        specs = await websocket.receive_json()
        logger.info(f"New websocket message from /shelve:\n    {'\n    '.join([f'{key}: {specs[key]}' for key in specs.keys()])}")
        book_data = format_db_record_as_book(DB.fetchone("""SELECT * FROM books WHERE uuid = ? """, (specs["uuid"],)))
        book_data.room = specs["room"]

        rooms = get_rooms()
        room = rooms[specs["room"]]

        suggested_shelf = None

        logger.info("Building list of viable shelves")
        try:
            position, neighbour = suggest_position(book_data, {specs["shelf"]: room.shelves[specs["shelf"]]}, DB)
            suggested_shelf = specs["shelf"]
        except CouldNotShelveError:
            pass

        shelves_list = []
        for shelf in list(room.shelves.values())[::-1]:
            try:
                _position, _neighbour = suggest_position(copy.copy(book_data), {shelf.uuid: room.shelves[shelf.uuid]}, DB)
                shelves_list.append({"name": shelf.name, "uuid": shelf.uuid, "disabled": False})
                if suggested_shelf is None:
                    suggested_shelf = shelf.uuid
                    position = _position
                    neighbour = _neighbour
            except CouldNotShelveError:
                shelves_list.append({"name": shelf.name, "uuid": shelf.uuid, "disabled": True})

        logger.info(
            f"{book_data.title} can be shelved on {len([shelf for shelf in shelves_list if not shelf['disabled']])} out of a possible {len(room.shelves)}."
        )

        response = {"neighbour": neighbour, "natlangpos": position.natlangpos, "shelves": shelves_list[::-1], "shelf": suggested_shelf}

        logger.info(f"Response: \n    {pprint.pformat(response, indent=4, width=140)}")

        await websocket.send_json(response)
