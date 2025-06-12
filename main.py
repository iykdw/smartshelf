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
from typing import Union
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

persist_dir = "/storage"

default_user_name = "foo"

class Book(BaseModel):
    isbn: str
    title: str
    subtitle: str
    author: str
    pages: int
    width: int
    room: str
    shelf: int
    position: int
    withdrawn: str
    time: Optional[str] = Field(default="0")
    user: Optional[str] = Field(default="")
    json: Optional[dict] = Field(default={})
    natlang_position: Optional[str] = Field(default="")


class Shelf(BaseModel):
    value: int|str
    width: int


class Room(BaseModel):
    name: str
    value: str
    shelves: List[Shelf]


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
    book: Book, room: Room, requested_shelf: int = -1
) -> Tuple[Book, str]:
    shelves = room.shelves.copy()

    if requested_shelf != -1:
        req_shelf = [shelves[requested_shelf]]
        del shelves[requested_shelf]
        shelves = req_shelf + shelves

    for i, shelf in enumerate(shelves):
        books_on_shelf = db_fetchall(
            """SELECT title, width, position FROM books WHERE room = ? AND shelf = ? ORDER BY position""",
            (
                room.value,
                shelf.value,
            ),
        )
        empty: List[int] = list(range(shelf.width))
        logging.info(
            f"Attempting to shelve {book.title} on shelf {shelf.value} in room {room.name}"
        )

        for shelved_book in books_on_shelf:
            hw = shelved_book[1] / 2  # halved width
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
        book.shelf = shelf.value
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


def db_fetchone(command: str, args: tuple = ()) -> List:
    return _db_execute(command, args, 1)


def db_fetchall(command: str, args: tuple = ()) -> List:
    return _db_execute(command, args, 2)


def get_rooms() -> Dict[str, Room]:
    with open(f"{persist_dir}/rooms.json", "r") as f:
        room_data = json.loads(f.read())

    rooms: Dict[str, Room] = {}

    for room in room_data:
        this_room = Room(
            name=room["name"],
            value=room["value"],
            shelves=[
                Shelf(width=shelf["width"], value=shelf["value"])

                for shelf in room["shelves"]
            ],
        )
        rooms[room["value"]] = this_room

    return rooms


def format_book_for_db_insertion(book: Book) -> Tuple[Union[str, int]]:
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


def format_db_record_as_book(record: Tuple[str | int]) -> Book:
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


@app.get("/", response_class=HTMLResponse)
def get_library(request: Request):
    user_name = request.headers.get('Remote-Name')
    if user_name is None:
        user_name = default_user_name

    books_raw = db_fetchall("""SELECT * FROM books""")

    books = []

    for book in books_raw:
        record_as_book = format_db_record_as_book(book)
        
        rooms = get_rooms()
        room = record_as_book.room
        if room == "":
            room = next(iter(rooms))
            
        
        shelf_width = rooms[room].shelves[int(record_as_book.shelf)].width
        pos = NatLangFractions(record_as_book.position, shelf_width, margin=0.05)
        if pos is not None:
            record_as_book.natlang_position = pos

        record_as_book.json = record_as_book.dict()
        books.append(record_as_book)

    ws_address = f"wss://{str(request.url).split('/')[2]}/search"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"user": user_name, "books": books, "ws_address": ws_address},
    )

class WithdrawRequest(BaseModel):
    isbn: str
    user_name: str

@app.post("/withdraw")
def withdraw(req: WithdrawRequest):
    print(req.isbn)
    book_data = format_db_record_as_book(
        db_fetchone("""SELECT * FROM books WHERE isbn = ? """, (req.isbn,))
    )

    book_data.time = time.time()
    book_data.user = req.user_name
    book_data.withdrawn = "withdrawn"
    response = update_book(book_data)
    print(response)
    return JSONResponse(content=jsonable_encoder(response))
    

@app.get("/book/{isbn}", response_class=HTMLResponse)
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
        request=request, name="book.html", context={"book": book_data}
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
            "book": book_data,
            "rooms": get_rooms().keys(),
            "mm_per_page": mm_per_page,
        },
    )


@app.post("/update")
def update_book(book: Book):
    print(book)

    if book.time != "0":
        time = round(float(book.time))

        if book.withdrawn == "withdrawn":
            db_execute(
                """INSERT INTO transactions VALUES (?, ?, ?, ?, ?)""",
                (book.isbn, "withdrawn", 0, time, book.user),
            )

            book.withdrawn = (
                f"{datetime.fromtimestamp(time).strftime('%Y-%m-%d')} by {book.user}"
            )

        elif book.withdrawn == "shelved":
            db_execute(
                """INSERT INTO transactions VALUES (?, ?, ?, ?, ?)""",
                (book.isbn, "shelved", 1, time, book.user),
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


@app.get("/shelve/{isbn}", response_class=HTMLResponse)
async def shelve(isbn: str, request: Request):
    book = format_db_record_as_book(
        db_fetchone("""SELECT * FROM books WHERE isbn = ? """, (isbn,))
    )

    rooms = get_rooms()
    ws_address = f"wss://{str(request.url).split('/')[2]}/shelve"
    book.withdrawn = ""

    first_room = next(iter(rooms))  # gets the first room
    suggest_vals = suggest_position(book, rooms[first_room])
    book_data = suggest_vals[0]
    neighbour = suggest_vals[1]
    room_list = [rooms[key] for key in rooms.keys()]

    return templates.TemplateResponse(
        request=request,
        name="shelve.html",
        context={
            "book": book_data,
            "neighbour": neighbour,
            "ws_address": ws_address,
            "rooms": room_list,
            "shelves": range(len(rooms[first_room].shelves)),
            "time": time.time(),
        },
    )


    


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
