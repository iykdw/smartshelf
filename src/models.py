from pydantic import BaseModel


class Book(BaseModel):
    uuid: str
    isbn: str
    title: str
    subtitle: str
    author: str
    pages: int
    width: int
    room: str
    shelf: str
    shelf_name: str
    position: int
    withdrawn: str
    time: str
    user: str
    natlangpos: str


class Shelf(BaseModel):
    uuid: str
    room: str
    name: str
    width: int


class Room(BaseModel):
    uuid: str
    name: str
    shelves: dict[str, Shelf]
