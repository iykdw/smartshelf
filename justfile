build:
    docker build . -t smartshelf

run: build
    docker run -v "$PWD/storage":/storage -p 8085:8085 --rm smartshelf
