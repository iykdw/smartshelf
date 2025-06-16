# smartshelf

Smartshelf is a home library management system, using Python and FastAPI and deployed using Docker.

Books can be added to the library by loading `/book/isbn`, where `isbn` is the <a href="https://en.wikipedia.org/wiki/ISBN">ISBN</a> of the book. Smartshelf will prepopulate the book's data from the Google Books API for you to approve before being added.

Books enrolled in Smartshelf can have one of two states - `shelved` or `withdrawn`. If a book is `shelved`, the web UI can be used to locate it on your shelves, to withdraw it to read. When you've finished reading it, Smartshelf will find the best position for it on your shelves. This might not be the same place it was shelved before - if you've added books in the meantime, or someone else has withdrawn books, there might be a gap which better suits the book size.

## Tech Stack

Smartshelf is developed using FastAPI and Pydantic alongside an SQLite3 database.

## Development & Deployment

To run locally: `just run`

To build the Docker container: `just build`

To build and then run the docker container: `just rebuild`
