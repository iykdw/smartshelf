CREATE TABLE books (
                uuid TEXT UNIQUE,
                isbn TEXT,
                title TEXT,
                subtitle TEXT,
                author TEXT,
                pages INTEGER,
                room TEXT,
                shelf TEXT,
                position INTEGER,
                width INTEGER,
                withdrawn TEXT) STRICT;
CREATE TABLE transactions (
           isbn TEXT,
           event TEXT,
           shelved INTEGER,
           time INTEGER UNIQUE,
           user TEXT) STRICT;
CREATE TABLE users (
            name TEXT UNIQUE) STRICT;
CREATE TABLE rooms (
        id TEXT UNIQUE,
        name TEXT) STRICT;
CREATE TABLE shelves (
        id TEXT UNIQUE,
        name TEXT,
        width INT,
        room TEXT,
        xloc INTEGER,
        yloc INTEGER) STRICT;
