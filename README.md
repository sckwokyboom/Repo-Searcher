---
title: Repo Searcher
emoji: 📈
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
license: mit
---

# Repo Searcher

## Running

### Local

1. Clone repository
2. Go to repository folder
3. Run `uv sync`
4. Run `uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 7860`

### Docker

1. Run `docker build -t repo-searcher .`
2. Run `docker run -p 7860:7860 repo-searcher`
