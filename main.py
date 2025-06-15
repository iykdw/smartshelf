import json
import logging
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from urllib.request import urlopen

from fastapi import FastAPI
from fastapi import Request
from fastapi import status
from fastapi import WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pydantic import Field

from natlangfractions import NatLangFractions

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

mm_per_page = 0.0696729243

persist_dir = "storage"

default_user_name = "foo"
ws_proto = "ws"

class CouldNotShelveError(Exception):
    pass

class Book(BaseModel):
    isbn: str
    title: str
    subtitle: str
    author: str
    pages: int
    width: int
    room: str
    shelf: str
    position: int
    withdrawn: str
    shelf_name: Optional[str] = Field(default="")
    time: Optional[int] = Field(default="0")
    user: Optional[str] = Field(default="")
    json: Optional[dict] = Field(default={})
    natlang_position: Optional[str] = Field(default="")

type BookTuple = Tuple[str, str, str, str, int, str, str, int, int, str] 

class Shelf(BaseModel):
    id: str
    name: str
    width: int


class Room(BaseModel):
    name: str
    value: str
    shelves: dict[str, Shelf]


def get_data_from_gb(isbn) -> Book:
    logging.info(f"Searching Google Books for ISBN {isbn}.")
    isbn = "".join([char for char in isbn if char.isdigit()])

    if len(isbn) not in [10, 13]:
        logging.info(f"Submitted string {isbn} isn't a valid ISBN.")
        # TODO: raise an error properly

    logging.info("Requesting data from Google Books.")
    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}&projection=full"
    response = urlopen(url)
    logging.info("Response received.")
    book_data = json.load(response)

    if book_data["totalItems"] == 0:
        return Book(
            isbn=isbn,
            title="",
            subtitle="",
            author="",
            pages=0,
            width=0,
            room="",
            shelf=-1,
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
        shelf=-2,
        position=-1,
        withdrawn="",
    )

    logging.info("Data read, returning now.")

    return book


def suggest_position(
    book: Book, room: Room, requested_shelf: str = ""
) -> Tuple[Book, str]:
    shelves = []

    if requested_shelf != "":
        req_shelf = room.shelves[requested_shelf]
        shelves.append(req_shelf)
        del room.shelves[requested_shelf]
        shelves += list(room.shelves.values())
    else:
        shelves = list(room.shelves.values())


    for shelf in shelves:
        books_on_shelf: List[Tuple[str, int, int]] = db_fetchall( #type:ignore because I know that this is fine
            """SELECT title, width, position FROM books WHERE room = ? AND shelf = ? ORDER BY position""",
            (
                room.value,
                shelf.id,
            ),
        )
        empty: List[int] = list(range(shelf.width))
        logging.info(
            f"Attempting to shelve {book.title} on shelf {shelf.value} in room {room.name}"
        )

        for shelved_book in books_on_shelf:
            hw  = int(shelved_book[1] / 2) # halved width
            start = int(shelved_book[2] - hw)
            end = int(shelved_book[2] + hw)
            logging.info(f"    {shelved_book[0]} in position {start}-{end}")

            if start in empty:
                curr = start  # remove the millimeters occupied by this book
            else:
                curr = start + 1

            print(empty)
            print(curr)
            print()

            while curr <= end:
                try:
                    empty.remove(curr)
                except ValueError:
                    logging.info("that fucking empty.remove(curr) bug, bestie, fix it")

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
                print(gaps)

        if len(gaps) == 0:
            logging.info("    Shelf completely full; proceeding to next shelf")

            continue

        gaps[-1].append(empty[-1])
        logging.info(f"    Gaps found - {gaps}")

        gaps = sorted(gaps, key=lambda x: x[1] - x[0], reverse=True)
        # Find the largest gap
        logging.info(f"    The largest gap is {gaps[0][1]-gaps[0][0]}mm")

        if (gaps[0][1] - gaps[0][0]) < book.width:
            # if the biggest gap is too small, proceed to next shelf
            logging.info("    Book is too wide to shelve; proceeding to next shelf")

            continue

        suggested = gaps[0][0] + int(book.width / 2)
        book.shelf = shelf.id
        book.position = suggested
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


def log_trace(statement):
    logging.info(statement)


def _db_execute(command: str, args: tuple, com_type: int) -> Optional[List]:
    with closing(sqlite3.connect(f"{persist_dir}/books.db")) as connection:
        connection.set_trace_callback(log_trace)
        with closing(connection.cursor()) as c:
            if com_type == 2:
                return c.execute(command, args).fetchall()
            elif com_type == 1:
                return c.execute(command, args).fetchone()
            else:
                c.execute(command, args)

            connection.commit()


def db_execute(command: str, args: tuple = ()) -> None:
    logging.info(command)
    logging.info(args)
    _db_execute(command, args, 0)


def db_fetchone(command: str, args: tuple = ()) -> Tuple:
    retval = _db_execute(command, args, 1)
    if retval is not None:
        return retval #type: ignore
    return [] #type: ignore

def db_fetchall(command: str, args: tuple = ()) -> Tuple:
    retvals = _db_execute(command, args, 2)
    if retvals is not None:
        return retvals #type: ignore
    return [] #type: ignore

def get_rooms() -> Dict[str, Room]:
    with open(f"{persist_dir}/rooms.json", "r") as f:
        room_data = json.loads(f.read())

    rooms: Dict[str, Room] = {}

    for room in room_data:
        shelf_ids = room["shelves"].keys()
        shelves = {}
        for shelf_id in shelf_ids:
            shelves[shelf_id] = Shelf(id=shelf_id, width=room["shelves"][shelf_id]["width"], name=room["shelves"][shelf_id]["name"])
        this_room = Room(
            name=room["name"],
            value=room["value"],
            shelves=shelves
        )
        rooms[room["value"]] = this_room

    return rooms


def format_book_for_db_insertion(book: Book) -> BookTuple:
    return_data = (
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


def format_db_record_as_book(record: BookTuple) -> Book:
    return Book(
        isbn=record[0],
        title=record[1],
        subtitle=record[2],
        author=record[3],
        pages=record[4],
        room=record[5],
        shelf=record[6],
        position=record[7],
        width=record[8],
        withdrawn=record[9],
    )


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f"{persist_dir}/debug.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
logger.info("Hello, world!")


logger.info("Checking db...")
db_execute(
    """CREATE TABLE IF NOT EXISTS books (
                isbn TEXT UNIQUE,
                title TEXT,
                subtitle TEXT,
                author TEXT,
                pages INTEGER,
                room TEXT,
                shelf INTEGER,
                position INTEGER,
                width INTEGER,
                withdrawn TEXT)
            STRICT""",
    (),
)
db_execute(
    """CREATE TABLE IF NOT EXISTS transactions (
           isbn TEXT,
           event TEXT,
           shelved INTEGER,
           time INTEGER UNIQUE,
           user TEXT)
           STRICT""",
    (),
)
db_execute(
    """CREATE TABLE IF NOT EXISTS users (
            name TEXT) STRICT""",
    (),
)
logger.info("DB check complete.")

"""
for shelf in list(get_rooms()["living"].shelves.values()):
    db_execute("UPDATE books SET shelf = ? WHERE shelf=?;", (shelf.id, shelf.name))
"""

@app.get("/", response_class=HTMLResponse)
@app.get("/{isbn}", response_class=HTMLResponse)
def get_library(request: Request, isbn: Optional[str] = ""):
    user_name = request.headers.get('Remote-Name')
    if user_name is None:
        user_name = default_user_name

    books_raw = db_fetchall("""SELECT * FROM books""")[::-1]

    books = []

    for book in books_raw:
        book = format_db_record_as_book(book)
        
        rooms = get_rooms()
        room = book.room
        if room == "":
            room = next(iter(rooms))

        if book.shelf == "" or book.position == -1:
            logger.info(f"Could not calculate position of book {book.title} on shelf {book.shelf} at position {book.position}.")
        else:
            shelf_width = rooms[room].shelves[book.shelf].width
            pos = NatLangFractions(book.position, shelf_width, margin=0.05)
            if pos is not None:
                book.natlang_position = pos

            book.shelf_name = rooms[book.room].shelves[book.shelf].name

        book.json = book.dict()
        books.append(book)

    if isbn != "":
        book_data = db_fetchone("""SELECT * FROM books WHERE isbn = ?""", (isbn,))

        if not book_data:
            book = get_data_from_gb(isbn)
        else:
            book = format_db_record_as_book(book_data)
        book.json = book.dict()
    else:
        book = Book(isbn="", title="", subtitle="", author="", pages=0, width=0, room="", shelf="", position=0, withdrawn="")

    ws_address = f"{ws_proto}://{str(request.url).split('/')[2]}/search"

    return templates.TemplateResponse(
        request=request, 
        name="index.html",
        context={"user": user_name, "select_book": book, "books": books, "ws_address": ws_address},
    )

class WithdrawRequest(BaseModel):
    isbn: str
    user_name: str

class ShelveRequest(BaseModel):
    isbn: str
    shelf: str

@app.post("/withdraw")
def withdraw(req: WithdrawRequest):
    book = format_db_record_as_book(
        db_fetchone("""SELECT * FROM books WHERE isbn = ? """, (req.isbn,))
    )

    logger.info(book.isbn , req.user_name)
    db_execute(
        """INSERT INTO transactions VALUES (?, ?, ?, ?, ?)""",
        (book.isbn, "withdrawn", 0, int(time.time()), req.user_name),
    )

    book.withdrawn = (
        f"{datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')} by {req.user_name}"
    )

    book.shelf = ""
    book.position = -1

    
    db_execute(
        """INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        format_book_for_db_insertion(book),
    )

    return JSONResponse(content=jsonable_encoder(book.withdrawn))
    
@app.post("/shelve")
async def shelve(req: ShelveRequest, request: Request):
    book = format_db_record_as_book(
        db_fetchone("""SELECT * FROM books WHERE isbn = ? """, (req.isbn,))
    )

    rooms = get_rooms()
    ws_address = f"{ws_proto}://{str(request.url).split('/')[2]}/shelve"
    book.withdrawn = ""
    
    if book.room != "":
        req_room = book.room
    else:
        first_room = next(iter(rooms))  # gets the first room
        req_room = first_room

    suggest_vals = suggest_position(book, rooms[req_room])
    book_data = suggest_vals[0]
    neighbour = suggest_vals[1]
    room_list = [rooms[key] for key in rooms.keys()]

    context={
        "book": book_data,
        "neighbour": neighbour,
        "ws_address": ws_address,
        "rooms": room_list,
        "shelves": len(rooms[req_room].shelves),
        "time": time.time(),
    }

    return JSONResponse(content=jsonable_encoder(context))

@app.get("/{isbn}", response_class=HTMLResponse)
def hit_endpoint(request: Request, isbn: str):
    book_data = db_fetchone("""SELECT * FROM books WHERE isbn = ?""", (isbn,))

    if not book_data:
        book_data = get_data_from_gb(isbn)
 
        db_execute(
            """INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            format_book_for_db_insertion(book_data),
        )

        return RedirectResponse(url=f"/edit/{isbn}")

    book_data = format_db_record_as_book(book_data)

    return templates.TemplateResponse(
            request=request, name="index.html", context={"book": book_data}
    )


@app.get("/edit/{isbn}", response_class=HTMLResponse)
def edit_book(request: Request, isbn: str):
    book_data = format_db_record_as_book(
        db_fetchone("""SELECT * FROM books WHERE isbn = ? """, (isbn,))
    )

    return templates.TemplateResponse(
        request=request, 
        name="validate.html",
        context={
            "book": book_data ,
            "rooms": get_rooms().keys(),
            "mm_per_page": mm_per_page,
        },
    )


@app.post("/update")
def update_book(book: Book):
    if book.time: 
        if book.withdrawn ==  "withdrawn":
            logger.info(book.isbn , book.user)
            db_execute(
                """INSERT INTO transactions VALUES (?, ?, ?, ?, ?)""",
                (book.isbn, "withdrawn", 0, int(time.time()), book.user),
            )

            book.withdrawn = (
                f"{datetime.fromtimestamp(float(book.time)).strftime('%Y-%m-%d')} by {book.user}"
            )

        elif book.withdrawn == "shelved":
            db_execute(
                """INSERT INTO transactions VALUES (?, ?, ?, ?, ?)""",
                (book.isbn, "shelved", 1, book.time, book.user),
            )

            book.withdrawn = ""

    db_execute(
        """INSERT OR REPLACE INTO books VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        format_book_for_db_insertion(book),
    )

    if book.shelf == -2:
        return RedirectResponse(
            url=f"/shelve/{book.isbn}", status_code=status.HTTP_303_SEE_OTHER
        )

    if book.time != "0":
        print(book.withdrawn)
        return book.withdrawn


@app.get("/locate/{isbn}", response_class=HTMLResponse)
def locate(isbn: str, request: Request):
    book_data = format_db_record_as_book(
            db_fetchone("""SELECT * FROM books WHERE isbn = ? """, (isbn,))
    )

    return templates.TemplateResponse(
        request=request,
        name="locate.html",
        context={"book": book_data},
    )


@app.websocket("/search")
async def search_websocket(websocket: WebSocket):
    await websocket.accept()

    while True:
        query = await websocket.receive_text()
        wild_query = f"%{query}%"
        results = db_fetchall(
            """SELECT isbn, title, author, withdrawn FROM books WHERE title LIKE ? OR subtitle LIKE ? OR author LIKE ?""",
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
            db_fetchone("""SELECT * FROM books WHERE isbn = ? """, (specs["isbn"],))
        )
        book_data.room = specs["room"]

        rooms = get_rooms()
        req_shelf = int(specs["shelf"])
        room = rooms[specs["room"]]

        results = suggest_position(book_data, room, requested_shelf=req_shelf)

        shelf = results[0].shelf
        neighbour = results[1]
        shelves = list(range(len(room.shelves)))

        await websocket.send_json(
            {
                "neighbour": neighbour,
                "shelves": shelves,
                "shelf": shelf,
            }
        )

@app.get("/book/{isbn}")
def redirect(isbn: str):
    return RedirectResponse(
        url=f"/{isbn}", status_code=status.HTTP_303_SEE_OTHER
    )
