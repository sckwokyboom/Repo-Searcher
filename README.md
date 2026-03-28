# Repo Searcher

## Как можно протестировать?
1. Перейти на https://aaxelis-repo-searcher.hf.space/
2. Выбрать один из проиндексированных репозиториев (под поисковой строкой перечислены)
3. На открывшейся странице ввести любой вопрос
4. Под поисковой строкой появятся найденные методы


## Training

Training code is located at `benchmark/lora_training/query_rewriter/train.py`
and here: `benchmark/lora_training/train_scorer.py` for LLM-MCTS-scorer.

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

### Local - cpu

1. Clone repository
2. Go to repository folder
3. Run `uv sync --extra cpu`
4. Run `uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 7860`

### Local - cuda

1. Clone repository
2. Go to repository folder
3. Run `uv sync --extra cu130`
4. Run `uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 7860`

Supported cudas: `cu130`, `cu128`, `cu126`

### Docker

1. Run `docker build -t repo-searcher .`
2. Run `docker run -p 7860:7860 repo-searcher`

