from typing import Optional

from pydantic import BaseModel
from pydantic import Field

class Book(BaseModel):
    isbn: str 
    title: str
    subtitle: str
    author: str
    pages: int
    width: int
    room: str
    shelf: str
    shelf_name: Optional[str] = Field(default="")
    position: int
    withdrawn: str
    time: Optional[str] = Field(default="0")
    user: Optional[str] = Field(default="")
    natlangpos: Optional[str] = Field(default="")


class Shelf(BaseModel):
    uuid: str
    room: str
    name: str
    width: int


class Room(BaseModel):
    uuid: str
    name: str
