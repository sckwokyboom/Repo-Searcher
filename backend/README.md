# Repo-Searcher Backend

API-сервер для индексации и поиска по Java-репозиториям. Гибридный поиск: BM25 + FAISS (dense embeddings) + reranking.

Построен на FastAPI + PyTorch + HuggingFace Transformers + tree-sitter.

## Требования

- Python 3.11+
- pip

## Установка и запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Сервер запустится на http://localhost:8000

Документация API (Swagger): http://localhost:8000/docs

## Переменные окружения

Все переменные имеют префикс `CODEGRAPH_`. Можно задать в файле `.env` в корне `backend/`.

| Переменная                    | По умолчанию                  | Описание                        |
|-------------------------------|-------------------------------|---------------------------------|
| `CODEGRAPH_GITHUB_TOKEN`      | —                             | GitHub API токен (опционально)  |
| `CODEGRAPH_FRONTEND_URL`      | `http://localhost:5173`       | URL фронтенда (для CORS)       |
| `CODEGRAPH_UNIXCODER_MODEL`   | `microsoft/unixcoder-base`    | Модель для эмбеддингов          |
| `CODEGRAPH_QWEN_MODEL`        | `Qwen/Qwen2.5-Coder-1.5B`    | Модель для reranking            |
| `CODEGRAPH_BM25_TOP_K`        | `30`                          | Top-K для BM25                  |
| `CODEGRAPH_FAISS_TOP_K`       | `30`                          | Top-K для FAISS                 |
| `CODEGRAPH_RRF_K`             | `60`                          | Параметр K для RRF              |
| `CODEGRAPH_RERANKER_TOP_K`    | `5`                           | Top-K после reranking           |

## API эндпоинты

| Метод    | Путь                                 | Описание                              |
|----------|--------------------------------------|---------------------------------------|
| `GET`    | `/api/health`                        | Health check                          |
| `GET`    | `/api/repos/search`                  | Поиск Java-репозиториев на GitHub     |
| `GET`    | `/api/repos/indexed`                 | Список проиндексированных репозиториев |
| `POST`   | `/api/repos/index`                   | Запуск индексации репозитория         |
| `GET`    | `/api/repos/{repo_id}/status`        | Статус индексации                     |
| `DELETE` | `/api/repos/{repo_id}`               | Удаление проиндексированного репозитория |
| `POST`   | `/api/repos/{repo_id}/search`        | Поиск по коду в репозитории           |
| `GET`    | `/api/repos/{repo_id}/graph/{method_id}` | Граф вызовов метода              |
| `WS`     | `/api/ws/indexing/{repo_id}`         | WebSocket: прогресс индексации        |

## Структура проекта

```
app/
├── main.py                 # FastAPI-приложение, CORS
├── config.py               # Настройки (pydantic-settings)
├── api/                    # Роуты
│   ├── repos.py            #   Репозитории: поиск, индексация, удаление
│   ├── search.py           #   Поиск по коду
│   ├── graph.py            #   Граф вызовов
│   └── ws.py               #   WebSocket прогресса
├── models/                 # Pydantic-модели
│   ├── repo.py             #   RepoInfo, IndexingProgress
│   ├── search.py           #   CodeChunk, SearchRequest/Response
│   └── graph.py            #   CallGraphNode, CallGraphEdge
├── indexer/                # Пайплайн индексации
│   ├── orchestrator.py     #   Оркестрация шагов индексации
│   ├── cloner.py           #   Клонирование репозитория
│   ├── parser.py           #   Парсинг Java-кода (tree-sitter)
│   ├── bm25_builder.py     #   Построение BM25-индекса
│   ├── vector_builder.py   #   Генерация эмбеддингов
│   ├── callgraph_builder.py #  Построение графа вызовов
│   └── store.py            #   Сохранение данных
├── search/                 # Поиск
│   ├── hybrid_retriever.py #   BM25 + FAISS + RRF + reranking
│   ├── query_expander.py   #   Расширение запроса
│   ├── graph_expander.py   #   Обход графа вызовов
│   └── reranker.py         #   Переранжирование результатов
└── ml/                     # ML-модели
    ├── model_manager.py    #   Загрузка и кэширование моделей
    └── unixcoder.py        #   UnixCoder для эмбеддингов
```

## Пайплайн

### Индексация
1. Клонирование репозитория с GitHub
2. Парсинг Java-файлов через tree-sitter (извлечение классов и методов)
3. Построение BM25-индекса (rank-bm25)
4. Генерация dense-эмбеддингов через UnixCoder
5. Построение FAISS-индекса
6. Построение графа вызовов (NetworkX)

### Поиск
1. Расширение запроса
2. Параллельный поиск: BM25 + FAISS
3. Объединение результатов через Reciprocal Rank Fusion (RRF)
4. Reranking через Qwen 2.5-Coder
5. Расширение результатов через граф вызовов

## Хранение данных

```
data/
├── repos/              # Клонированные репозитории
└── indexes/            # Индексы
    ├── registry.json   # Реестр проиндексированных репозиториев
    └── {repo_id}/      # Данные конкретного репозитория
        ├── metadata.json
        ├── chunks.pkl
        ├── bm25_index/
        ├── vectors.npy
        ├── faiss.index
        └── callgraph.gml
```
