FROM node:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . ./app

# Compile frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

WORKDIR /app

RUN uv sync

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
