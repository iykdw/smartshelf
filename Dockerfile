FROM ghcr.io/astral-sh/uv:python3.13-alpine


# Copy the project into the image
RUN mkdir /app
COPY pyproject.toml /app
COPY uv.lock /app

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv sync --locked --compile-bytecode

ADD . /app

CMD ["uv", "run", "uvicorn", "main:app", "--proxy-headers", "--forwarded-allow-ips=*", "--host", "0.0.0.0", "--port", "8085"]
