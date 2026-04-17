FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
COPY packages/ packages/
COPY etl/ etl/
COPY tools/qafiya_annotator/ tools/qafiya_annotator/

RUN uv pip install --system \
    pydantic \
    sqlalchemy \
    psycopg[binary] \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    && uv pip install --system -e packages/schema \
    && uv pip install --system -e etl

EXPOSE 8501

ENV DATABASE_URL=postgresql+psycopg://alshaer:alshaer@postgres:5432/alshaer

CMD ["uvicorn", "tools.qafiya_annotator.backend:app", "--host", "0.0.0.0", "--port", "8501", "--reload"]
