build:
    docker build . -t smartshelf

test:
    uv run pytest --cov=. --cov-report=html:missing --cov-report=term tests/

run:
    uv run uvicorn --app-dir src main:app --proxy-headers --forwarded-allow-ips=* --host 0.0.0.0 --port 8085

rebuild: build run
   docker run -p 8085:8085 --rm smartshelf

publish: build
    docker tag smartshelf praxidyke/smartshelf:latest
    docker push praxidyke/smartshelf
