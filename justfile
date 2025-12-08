build:
    docker build . -t smartshelf

run:
    uv run uvicorn main:app --proxy-headers --forwarded-allow-ips=* --host 0.0.0.0 --port 8085

rebuild: build run
   docker run -p 8085:8085 --rm smartshelf
