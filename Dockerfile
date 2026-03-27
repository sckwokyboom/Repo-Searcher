FROM node:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# # Install nvm
# RUN wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash

# # Install node
# RUN nvm install --lts

# Install uv
# RUN wget -qO- https://astral.sh/uv/0.11.1/install.sh | sh

COPY . ./app

# Compile frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

WORKDIR /app

RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
