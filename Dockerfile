FROM python:3.12

WORKDIR /app

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock ./

RUN poetry install

COPY . .

CMD ["poetry", "run", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "30", "--timeout-graceful-shutdown", "600", "--loop", "uvloop", "--limit-concurrency", "500"]