# smartshelf

## Development & Deployment

1) `docker build . -t smartshelf`
2) `docker run -p 8085:8000 --rm smartshelf` or use `docker-compose.yaml`

## Developing

Replace Step 2 with:

2) `docker run -d -v "$PWD/storage":/storage -p 8085:8085 --rm smartshelf`
