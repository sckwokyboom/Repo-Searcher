FROM alpine:3.23.3

# Install nvm
RUN wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | sh

# Install node
RUN nvm install --lts

# Install uv
RUN wget -qO- https://astral.sh/uv/0.11.1/install.sh | sh

COPY . ./app

WORKDIR /app

RUN uv sync

CMD sh ./start_docker.sh