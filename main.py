import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from typing import Annotated
from typing import List
from typing import Tuple
from urllib.request import urlopen
from uuid import uuid4

from fastapi import FastAPI
from fastapi import Form
from fastapi import Request
from fastapi import status
from fastapi import WebSocket
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import db_ops
from models import Book
from models import Room
from models import Shelf
from natlangposition import nat_lang_position
from rooms_to_db import rooms_to_db


class CouldNotShelveError(Exception):
    pass

class DatabaseNotFoundError(Exception):
    pass

class UnconfiguredError(Exception):
    pass

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

def get_data_from_gb(isbn) -> Book:
    if len(isbn) not in [10, 13]:
        logging.info(f"Submitted string {isbn} isn't a valid ISBN.")
        # TODO: raise an error properly

    logging.info(f"Searching Google Books for ISBN {isbn}.")
    isbn = "".join([char for char in isbn if char.isdigit()])

    logging.info("Requesting data from Google Books.")
    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}&projection=full"
    response = urlopen(url)
    logging.info("Response received.")
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
    logging.info(f"Book has ID {book_id}, searching for that.")

    url = f"https://content-books.googleapis.com/books/v1/volumes/{book_id}"
    response = urlopen(url)
    logging.info("Response received.")

    book_data = json.load(response)["volumeInfo"]
    book = Book(
        uuid=str(uuid4()),
        isbn=isbn,
        title=book_data["title"],
        subtitle=book_data["subtitle"] if "subtitle" in book_data else "",
        author=", ".join(book_data["authors"]) if "authors" in book_data else "",
        pages=book_data["printedPageCount"] if "printedPageCount" in book_data else 0,
        width=int(
            (
                int(book_data["printedPageCount"])

                if "printedPageCount" in book_data
                else 0
            )
            * mm_per_page
        ),
        room="",
        shelf="",
        position=-1,
        withdrawn="",
    )

    logging.info(f"Book {book.title} has been assigned uuid {book.uuid}.")
    logging.info("Data read, returning now.")

    return book

def suggest_position(
        book: Book, room_shelves: dict[str, Shelf], DB: db_ops.DB, requested_shelf: str = ""
) -> Tuple[Book, str]:
    shelves = []

    if requested_shelf != "":
        req_shelf = room_shelves[requested_shelf]
        shelves.append(req_shelf)
        del room_shelves[requested_shelf]
        shelves += list(room_shelves.values())
    else:
        shelves = list(room_shelves.values())

    for shelf in shelves:
        books_on_shelf: List[Tuple[str, int, int]] = DB.fetchall(
            """SELECT title, width, position FROM books WHERE shelf = ? ORDER BY position""",
            (
                shelf.uuid,
            ),
        )
        empty: List[int] = list(range(shelf.width))
        logging.debug(shelf)
        logging.debug(
                f"Attempting to shelve {book.title} on shelf {shelf.uuid}"
        )

        for shelved_book in books_on_shelf:
            hw = int(shelved_book[1] / 2) # halved width
            start = int(shelved_book[2] - hw)
            end = int(shelved_book[2] + hw)
            logging.info(f"    {shelved_book[0]} in position {start}-{end}")

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
                # If there are no gaps known, start a new gap at -1 because... offset reasons
                gaps.append([-1])
            elif len(gaps) == 0:
                gaps.append([empty[0]])

            if len(gaps[-1]) == 2:
                # If  all gaps are complete, start a new gap
                gaps.append([empty[j]])

            if empty[j] != empty[j + 1] - 1:
                # If the next empty millimetre is not the next millimetre, finish this gap.
                gaps[-1].append(empty[j])

        if len(gaps) == 0:
            logging.info("    Shelf completely full; proceeding to next shelf")

            continue

        gaps[-1].append(empty[-1])
        logging.debug(f"    Gaps found - {gaps}")

        gaps = sorted(gaps, key=lambda x: x[1] - x[0], reverse=True)
        # Find the largest gap
        largest = gaps[0][1]-gaps[0][0]
        logging.info(f"    The largest gap is {largest}mm. Book is {book.width}mm wide and would {'not ' if book.width > largest else ''}fit.")

        if (gaps[0][1] - gaps[0][0]) < book.width:
            # if the biggest gap is too small, proceed to next shelf
            logging.debug("    Book is too wide to shelve; proceeding to next shelf")

            continue

        suggested = gaps[0][0] + int(book.width / 2)
        book.shelf = shelf.uuid
        book.position = suggested
        book.natlangpos = nat_lang_position(book.position, shelf.width)

        logging.info(f"    Suggested position is {gaps[0][0]}")

        neighbour = "shelf edge"

        if gaps[0][0] == -1 and gaps[0][1]:
            return (
                book,
                neighbour,
            )

        else:
            neighbour = books_on_shelf[0][0]

            for shelved in books_on_shelf:

                if shelved[2] > book.position:
                    return (
                        book,
                        neighbour,
                    )
                neighbour = shelved[0]

        return (
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

def get_shelves(room: str = "") -> List[Shelf]:
    shelves = []
    if room == "":
        shelf_data = DB.fetchall("""SELECT * FROM shelves""")

    else:
        shelf_data = DB.fetchall("""SELECT * FROM shelves WHERE room = ?""", (room,))

    for s in shelf_data:
        shelves.append(Shelf(uuid=s[0], name=s[1], width=s[2], room=s[3]))

    return shelves


def format_book_for_db_insertion(book: Book) -> Tuple[str | int]:
    return_data =  (
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


def format_db_record_as_book(record: Tuple[str | int]) -> Book:
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
_test_env = False
mm_per_page = 0.0696729243


if "pytest" in sys.modules:
    _test_env = True
    persist_dir = "storage/test"
    db_file =  f"{datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')}.db"

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging. FileHandler(f"{persist_dir}/debug.log"),
        logging.StreamHandler(sys.stdout),
    ], 
)

logger = logging.getLogger(__name__)
logger.info("Hello, world!")


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DB = db_ops.DB(persist_dir, db_file, logging)

if _test_env:
    _build_db(DB)

_build_db(DB)

logger.info(f"Checking db at {persist_dir}/{db_file}...")
if has_config(DB):
    logger.info("DB check complete.")
else:
    raise DatabaseNotFoundError



@app.get("/", response_class=HTMLResponse)
def get_library(request: Request):
    if not has_config(DB):
        raise UnconfiguredError

    books_raw = DB.fetchall("""SELECT * FROM books""")[::-1]

    books = []

    for book_data in books_raw:
        books.append(format_db_record_as_book(book_data))

    ws_address = f"wss://{str(request.url).split('/')[2]}/search"

    return templates.TemplateResponse(
        request=request,
        name="library.html",
        context={"books": books, "ws_address": ws_address},
    )

@app.get("/populate/{isbn}")
def add_book(isbn: str):
    book_data = get_data_from_gb(isbn)

    DB.execute(
        """INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        format_book_for_db_insertion(book_data),
    )
    
    uuid = book_data.uuid
    return RedirectResponse(url=f"/book/{uuid}", status_code=status.HTTP_303_SEE_OTHER)
    

@app.get("/book/{uuid}", response_class=HTMLResponse)
def hit_endpoint(request: Request, uuid: str):
    book_data = DB.fetchone("""SELECT * FROM books WHERE uuid = ?""", (uuid,))
    logger.info(uuid)

    if not book_data: # If this book is being added
        return RedirectResponse(url=f"/add/{uuid}", status_code=status.HTTP_302_FOUND)

    book = format_db_record_as_book(book_data)
    return templates.TemplateResponse(
        request=request, name="book.html", context={"book": book}
    )

@app.get("/edit/{uuid}", response_class=HTMLResponse)
def edit_book(request: Request, uuid: str):
    book_data = format_db_record_as_book(
        DB.fetchone("""SELECT * FROM books WHERE uuid = ? """, (uuid,))
    )

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

            book.withdrawn = (
                f"{datetime.fromtimestamp(time).strftime('%Y-%m-%d')} by {book.user}"
            )

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
        return RedirectResponse(
            url=f"/shelve/{book.isbn}", status_code=status.HTTP_303_SEE_OTHER
        )

    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/shelve/{uuid}", response_class=HTMLResponse)
async def shelve(uuid: str, request: Request):
    book = format_db_record_as_book(
        DB.fetchone("""SELECT * FROM books WHERE uuid = ? """, (uuid,))
    )

    rooms = get_rooms()
    ws_address = f"wss://{str(request.url).split('/')[2]}/shelve"
    book.withdrawn = ""

    first_room = next(iter(rooms))  # gets the first room
    suggest_vals = suggest_position(book, rooms[first_room].shelves, DB)
    book_data = suggest_vals[0]
    neighbour = suggest_vals[1]
    room_list = [rooms[key] for key in rooms.keys()]
    room_options: List[List[str,str]] = []
    for room in room_list:
        room_options.append([room.uuid, room.name])

    shelves = [rooms[first_room].shelves[key] for key in rooms[first_room].shelves.keys()]
    return templates.TemplateResponse(
        request=request,
        name="shelve.html",
        context={
            "book": book_data,
            "neighbour": neighbour,
            "ws_address": ws_address,
            "rooms": room_options,
            "shelves": shelves,
            "time": time.time(),
        },
    )


@app.get("/withdraw/{uuid}", response_class=HTMLResponse)
def withdraw(uuid: str, request: Request):
    user = request.headers.get('Remote-Name')
    if user is None:
        user = default_user
    book_data = format_db_record_as_book(
        DB.fetchone("""SELECT * FROM books WHERE uuid = ? """, (uuid,))
    )

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
    book = format_db_record_as_book(
        DB.fetchone("""SELECT * FROM books WHERE uuid = ? """, (uuid,))
    )
    logging.info(str(book))

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

    logging.info(f"Adding room {room.name} with uuid {room.uuid}.")
    DB.execute("""INSERT INTO rooms VALUES (?, ?)""", (room.uuid, room.name,))

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

    logging.info(f"Adding shelf {shelf.name} with uuid {shelf.uuid} in room {room}.")
    DB.execute("""INSERT INTO shelves VALUES (?, ?, ?, ?)""", (shelf.uuid, shelf.name, int(shelf.width), shelf.room,))

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
        print(specs)
        book_data = format_db_record_as_book(
            DB.fetchone("""SELECT * FROM books WHERE uuid = ? """, (specs["uuid"],))
        )
        book_data.room = specs["room"]

        rooms = get_rooms()
        req_shelf = int(specs["shelf"])
        room = rooms[specs["room"]]

        results = suggest_position(book_data, room.shelves, DB, requested_shelf=req_shelf)

        shelf = results[0].shelf
        neighbour = results[1]
        shelves = list(range(len(room.shelves)))

        await websocket.send_json(
            {
                "neighbour": neighbour,
                "natlangpos": results[0].natlangpos,
                "shelves": shelves,
                "shelf": shelf,
            }
        )
