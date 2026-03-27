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

## Training

Training code is located at `benchmark/lora_training/query_rewriter/train.py`

## Testing

### Indexing

1. On main page input repository URL to the search bar.
2. From the dropdown, select desired repository.
3. Indexing process will start.
4. On indexing completion, new search bar will appear.
5. Search for desired code snippet.

### Searching

1. On main page select desired indexed repository under the search bar.
2. Search for desired code snippet.

## Running

### Local

1. Clone repository
2. Go to repository folder
3. Run `uv sync`
4. Run `uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 7860`

### Docker

1. Run `docker build -t repo-searcher .`
2. Run `docker run -p 7860:7860 repo-searcher`

