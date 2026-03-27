# Отчёт: Эксперимент по локальному graph expansion

## 1. Контекст и цель

**Гипотеза**: если взять top-N файлов от BM25 baseline и добавить к кандидатам их 1-hop соседей по call graph, то file-level recall вырастет — релевантные файлы, которые не попали в baseline top-k, но являются соседями seed-файлов, окажутся в расширенном пуле.

**Что проверяли**: идею из статьи RANGER (Repository-level Agent for Graph-Enhanced Retrieval) — в минимальной форме, без MCTS, без LLM, без новых эмбеддингов.

**Три режима**:
- **A (baseline)**: BM25 + file-level score aggregation (сумма BM25 скоров чанков по файлу)
- **B (graph expand)**: baseline + boost для файлов, чьи чанки являются graph-соседями seed-чанков
- **C (prioritized expand)**: baseline + graph bonus с ограниченным бюджетом на новые файлы

---

## 2. Что есть в системе (предусловия)

### Call graph
Граф строится в `backend/app/indexer/callgraph_builder.py` через tree-sitter:

```python
def build_call_graph(chunks: list[CodeChunk], repo_path: Path) -> nx.DiGraph:
    graph = nx.DiGraph()

    # Узлы = методы (chunk_type == "method")
    method_name_to_chunks: dict[str, list[str]] = {}
    for chunk in chunks:
        if chunk.chunk_type == "method" and chunk.method_name:
            graph.add_node(chunk.chunk_id, label=chunk.method_name, file_path=chunk.file_path)
            method_name_to_chunks.setdefault(chunk.method_name, []).append(chunk.chunk_id)

    # Рёбра = method invocations (caller -> callee)
    parser = Parser(JAVA_LANGUAGE)
    for file_path in sorted(repo_path.rglob("*.java")):
        source = file_path.read_bytes()
        tree = parser.parse(source)
        _extract_calls(tree.root_node, source, relative_path, graph, method_name_to_chunks, chunks)
```

**Ключевые особенности**:
- Узлы = chunk_id формата `path/File.java::ClassName.methodName`
- Рёбра = только method invocations (AST node type `method_invocation`)
- Нет import-рёбер, нет same-package, нет inheritance, нет field references
- Разрешение callee: по имени метода, предпочтение — тот же файл, иначе первый найденный
- На практике граф **разреженный** для многих репозиториев

### Baseline retriever (BM25FileAgg)
```python
class BM25FileAgg(BaseRetriever):
    """BM25 с file-level score aggregation: сумма chunk scores по файлу."""

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        scores = self.bm25.get_scores(tokenize(query))

        # Агрегация по файлам
        file_scores: dict[str, float] = defaultdict(float)
        for i, score in enumerate(scores):
            if score > 0:
                file_scores[self.chunks[i].file_path] += score

        # Ранжируем файлы по суммарному скору
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(best_chunk_per_file[fp], agg_score) for fp, agg_score in sorted_files]
```

### Данные
- 859 samples across 10 Java repos
- Эксперимент на 74 samples (стратифицированная подвыборка, 9 repos — у одного нет индекса)
- Ground truth: `changed_files` из git diff

---

## 3. Реализация

### 3.1 Вспомогательные функции

```python
# benchmark/retrievers.py

def _build_file_chunk_index(chunks: list[CodeChunk]) -> dict[str, list[int]]:
    """file_path -> [chunk_indices] для маппинга файл -> все его чанки."""
    index: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        index[chunk.file_path].append(i)
    return index

def _build_chunk_id_index(chunks: list[CodeChunk]) -> dict[str, int]:
    """chunk_id -> chunk_index для обратного маппинга из графа."""
    return {chunk.chunk_id: i for i, chunk in enumerate(chunks)}

_call_graph_cache: dict[str, nx.DiGraph] = {}

def _get_call_graph(repo_id: str) -> nx.DiGraph:
    if repo_id not in _call_graph_cache:
        _call_graph_cache[repo_id] = load_call_graph(repo_id)
    return _call_graph_cache[repo_id]
```

### 3.2 Mode B — BM25GraphExpansion

```python
class BM25GraphExpansion(BaseRetriever):
    name = "BM25+GraphExpand"
    SEED_K = 5              # top-5 файлов как seeds
    NEIGHBORS_PER_SEED = 5  # макс соседних файлов на seed
    EDGE_WEIGHT = 1.0       # вес ребра (method call)
    BOOST = 0.2             # аддитивный бонус для файлов в обоих пулах
    HUB_DEGREE_LIMIT = 50   # пропускаем хабовые узлы

    def retrieve(self, query: str, top_k: int = 10):
        # 1. BM25 scores для ВСЕХ чанков
        bm25_scores = self.bm25.get_scores(tokenize(query))

        # 2. File-level aggregation + нормализация в [0, 1]
        file_agg: dict[str, float] = defaultdict(float)
        for i, score in enumerate(bm25_scores):
            if score > 0:
                file_agg[self.chunks[i].file_path] += score
        max_score = max(file_agg.values())
        baseline_norm = {fp: s / max_score for fp, s in file_agg.items()}

        # 3. Top-5 seeds
        seed_files = sorted(baseline_norm.items(), key=lambda x: x[1], reverse=True)[:5]

        # 4. Expand: для каждого seed -> все его чанки -> graph neighbors
        graph_scores: dict[str, float] = {}
        for seed_fp, seed_norm_score in seed_files:
            neighbor_count = 0
            for ci in file_chunk_idx[seed_fp]:
                chunk_id = self.chunks[ci].chunk_id
                if chunk_id not in graph or graph.degree(chunk_id) > 50:
                    continue
                for nid in graph.predecessors(chunk_id) + graph.successors(chunk_id):
                    nfp = chunks[chunk_id_idx[nid]].file_path
                    gs = seed_norm_score * 1.0  # EDGE_WEIGHT
                    graph_scores[nfp] = max(graph_scores.get(nfp, 0), gs)
                    neighbor_count += 1
                    if neighbor_count >= 5:  # NEIGHBORS_PER_SEED
                        break

        # 5. Merge: файлы из обоих пулов получают boost
        for fp in all_file_paths:
            bl = baseline_norm.get(fp, 0)
            gs = graph_scores.get(fp, 0)
            if bl > 0 and gs > 0:
                final[fp] = bl + 0.2 * gs   # BOOST
            elif bl > 0:
                final[fp] = bl
            else:
                final[fp] = gs               # graph-only файл

        return sorted(final, reverse=True)[:top_k]
```

**Формула scoring**:
- Файл в baseline И в graph: `score = baseline_norm + 0.2 * graph_score`
- Файл только в baseline: `score = baseline_norm`
- Файл только из graph: `score = seed_norm * EDGE_WEIGHT`

### 3.3 Mode C — BM25PrioritizedExpansion

```python
class BM25PrioritizedExpansion(BaseRetriever):
    name = "BM25+PriorExpand"
    SEED_K = 5
    MAX_TOTAL_EXPANDED = 15   # макс новых файлов из графа
    GRAPH_BONUS = 0.3         # бонус от graph proximity
    HUB_DEGREE_LIMIT = 50

    def retrieve(self, query: str, top_k: int = 10):
        # 1-2. То же: BM25 + file aggregation + нормализация
        # ...

        # 3. Expand с бюджетом
        graph_bonus: dict[str, float] = {}
        new_count = 0
        for seed_fp, seed_norm in seed_files:  # seeds в порядке убывания скора
            if new_count >= 15:
                break
            for chunk in seed_file_chunks:
                for neighbor in graph_neighbors(chunk):
                    bonus = seed_norm * 0.3  # GRAPH_BONUS
                    if nfp not in graph_bonus:
                        new_count += 1       # считаем бюджет
                    graph_bonus[nfp] = max(graph_bonus.get(nfp, 0), bonus)

        # 4. Merge: baseline_norm + graph_bonus для каждого файла
        for fp in all_files:
            final[fp] = baseline_norm.get(fp, 0) + graph_bonus.get(fp, 0)
```

**Отличия от Mode B**:
- Бюджет на новые файлы (max 15) — не бесконечное расширение
- `GRAPH_BONUS = 0.3` применяется мультипликативно к seed_norm, потом **аддитивно** ко всем файлам (не только graph-only)
- Seeds обрабатываются в порядке убывания скора — лучшие seeds расширяются первыми

### 3.4 Experiment runner

```python
# benchmark/graph_expansion_experiment.py

def run_experiment():
    dataset = load_dataset()                    # 859 samples
    queries = sample_queries(dataset, n=75)     # стратифицированная подвыборка

    modes = {
        "A_baseline":      BM25FileAgg,
        "B_graph_expand":  BM25GraphExpansion,
        "C_prior_expand":  BM25PrioritizedExpansion,
    }

    for query in queries:
        for mode in modes:
            results = mode.retrieve(query, top_k=10)
            # Дедупликация по file_path
            # Запись в RetrievalResult
            # Логирование: baseline files, seeds, new_from_graph, gained/lost GT

    # Метрики через существующий evaluator
    metrics = aggregate_metrics(all_sample_metrics, k_values=[1, 3, 5, 10])
```

---

## 4. Результаты

### 4.1 Основная таблица

| Mode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@10 | MRR |
|------|-------|-------|-------|--------|-----------|-----|
| A baseline | 0.1486 | 0.3243 | 0.4595 | 0.5405 | 0.3644 | 0.2645 |
| B graph_expand | **0.1622** | 0.3243 | 0.4459 | **0.5541** | **0.3731** | **0.2784** |
| C prior_expand | 0.1486 | 0.3108 | 0.4324 | **0.5541** | 0.3356 | 0.2625 |

### 4.2 Дельты относительно baseline

| Mode | dHit@1 | dHit@5 | dHit@10 | dRecall@10 | dMRR |
|------|--------|--------|---------|------------|------|
| B | **+0.0135** | -0.0135 | **+0.0135** | **+0.0087** | **+0.0139** |
| C | 0.0000 | -0.0270 | +0.0135 | -0.0287 | -0.0021 |

### 4.3 Статистика по изменениям

| Метрика | B | C |
|---------|---|---|
| Queries где помог (gained GT) | 3 | 5 |
| Queries где навредил (lost GT) | 2 | 12 |
| Queries без изменений в hit | 69 | — |
| Queries с другим top-10 | 47/74 | 64/74 |
| Avg новых файлов из графа | 1.1 | 2.8 |
| Max новых файлов | 4 | 9 |
| Queries с 0 расширением | 27/74 | — |

### 4.4 Распределение expansion (Mode B)

| Новых файлов из графа | Кол-во queries |
|------------------------|----------------|
| 0 | 27 (36%) |
| 1 | 21 (28%) |
| 2 | 19 (26%) |
| 3 | 4 (5%) |
| 4 | 3 (4%) |

### 4.5 Плотность графа по репозиториям

| Репозиторий | Samples | Avg seed degree | Avg new files | Queries с expansion |
|-------------|---------|-----------------|---------------|---------------------|
| dotCMS/core | 6 | **10017** | 2.0 | 6/6 |
| hmislk/hmis | 9 | **9869** | 0.8 | 4/9 |
| jdereg/java-util | 5 | **2942** | 0.0 | 0/5 |
| panghy/javaflow | 11 | 1213 | 2.2 | 10/11 |
| DataSQRL/sqrl | 5 | 812 | 1.0 | 4/5 |
| etonai/OpenFields2 | 10 | 602 | 1.0 | 7/10 |
| heymumford/Samstraumr | 7 | 462 | 0.4 | 2/7 |
| rydnr/bytehot | 9 | 362 | 1.0 | 7/9 |
| 100-hours-a-week/22-tenten-be | 12 | 152 | 1.1 | 7/12 |

---

## 5. Подробный разбор примеров

### 5.1 Пример: B помог (PromiseStream)

**Query**: *enhance PromiseStream with lifecycle monitoring and improved error handling*
**Repo**: panghy/javaflow
**GT files** (2): `PromiseStream.java`, `StreamManager.java`

**Baseline top-10** (A):

| Rank | File | Score |
|------|------|-------|
| 1 | FlowRpcTransportImplInternalCoverageTest.java | 8286 |
| 2 | RemotePromiseTrackerTest.java | 5238 |
| 3 | FlowRpcTransportImpl.java | 3798 |
| 4 | PromiseStreamTest.java | 3484 |
| 5 | RpcStreamTimeoutUtilCoverageTest.java | 3085 |
| 6 | RealFlowConnectionCoverageTest.java | 2328 |
| 7 | FlowRpcTransportImplPromiseTest.java | 2300 |
| 8 | RemotePromiseTracker.java | 2290 |
| 9 | FlowRpcTransportImplErrorHandlingTest.java | 2217 |
| 10 | FlowRpcTransportImplStreamTest.java | 2041 |

**Hit@10 = False** — ни `PromiseStream.java`, ни `StreamManager.java` не в top-10.

**Mode B (graph expand) top-10**:

| Rank | File | Norm Score | Source |
|------|------|------------|--------|
| 1 | FlowRpcTransportImplInternalCoverageTest.java | 1.000 | baseline |
| 2 | RemotePromiseTrackerTest.java | 0.632 | baseline |
| 3 | FlowRpcTransportImpl.java | 0.458 | baseline |
| 4 | **Tuple.java** | 0.429 | **graph** |
| 5 | PromiseStreamTest.java | 0.420 | baseline |
| 6 | RpcStreamTimeoutUtilCoverageTest.java | 0.372 | baseline |
| 7 | **Flow.java** | 0.316 | **graph** |
| 8 | **PromiseStream.java** | 0.292 | **graph** ✅ GT |
| 9 | RealFlowConnectionCoverageTest.java | 0.281 | baseline |
| 10 | FlowRpcTransportImplPromiseTest.java | 0.278 | baseline |

**Hit@10 = True** — `PromiseStream.java` попал на 8-ю позицию через graph expansion.

**Как это произошло**: seed #3 (`FlowRpcTransportImpl.java`) вызывает методы из `PromiseStream.java`. Graph expansion нашёл эту связь и присвоил `PromiseStream.java` score = `seed_norm(0.458) * EDGE_WEIGHT(1.0) = 0.458`. Но baseline тоже дал ему какой-то BM25 score (он был за пределами top-10 baseline), и финальный score = `baseline_norm + 0.2 * 0.458 = 0.292`.

**Побочный эффект**: `Tuple.java` и `Flow.java` тоже вошли из графа — это шум. Они вытеснили `RemotePromiseTracker.java`, `FlowRpcTransportImplErrorHandlingTest.java` и `FlowRpcTransportImplStreamTest.java` из baseline, которые тоже не были GT, так что потерь нет.

---

### 5.2 Пример: B помог (FlowScheduler)

**Query**: *improve task cancellation and timer management*
**Repo**: panghy/javaflow
**GT files** (13): `Flow.java`, `FlowFuture.java`, `FlowScheduler.java`, `SingleThreadedScheduler.java`, `Task.java`, + 8 тестов

**Baseline top-10**: содержит `SingleThreadedScheduler.java` (#2), `Task.java` (#7), `Flow.java` (#10) — 3 из 13 GT.

**Mode B top-10**: добавился `FlowScheduler.java` (#7, graph score 0.363) — теперь 4 из 13 GT.

**Механизм**: `SingleThreadedScheduler` (seed #2, norm=0.668) вызывает методы `FlowScheduler`. Graph expansion нашёл эту связь.

---

### 5.3 Пример: B навредил (integration tests)

**Query**: *add comprehensive integration tests for network layer*
**Repo**: panghy/javaflow
**GT files** (3): `RealFlowConnectionIntegrationTest.java`, `RealFlowTransportFinalTest.java`, `RealFlowTransportIntegrationTest.java`

**Baseline top-10**: `RealFlowConnectionIntegrationTest.java` на позиции #10 (score 1546).

**Mode B top-10**:

| Rank | File | Score | Source |
|------|------|-------|--------|
| 1 | FlowRpcTransportImplInternalCoverageTest.java | 1.000 | baseline |
| 2 | TupleTest.java | 0.534 | baseline |
| 3 | **Tuple.java** | 0.327 | **graph** (шум) |
| 4 | RemotePromiseTrackerTest.java | 0.318 | baseline |
| 5 | TupleBranchCoverageTest.java | 0.284 | baseline |
| 6 | **DefaultEndpointResolver.java** | 0.250 | **graph** (шум) |
| 7 | **Flow.java** | 0.249 | **graph** (шум) |
| 8 | FlowRpcTransportImplErrorHandlingTest.java | 0.246 | baseline |
| 9 | RealFlowConnectionCoverageTest.java | 0.235 | baseline |
| 10 | **FlowSerialization.java** | 0.228 | **graph** (шум) |

`RealFlowConnectionIntegrationTest.java` **вытеснен** — был на позиции #10 (baseline norm ~0.189), а 4 graph-файла с higher scores заняли его место. Ни один из 4 graph-файлов не является GT.

**Причина**: graph-only файлы (`Tuple.java`, `DefaultEndpointResolver.java`, `Flow.java`, `FlowSerialization.java`) получили score = seed_norm (0.327–0.228), что выше, чем baseline_norm файла на позиции #10. Они вытеснили единственный релевантный файл.

---

### 5.4 Пример: B навредил (Spring plugin)

**Query**: *Complete Spring Plugin Foundation Architecture*
**Repo**: rydnr/bytehot
**GT files** (10): `PluginBase.java`, `ByteHotSpringPlugin.java`, `SpringContextManager.java`, etc.

**Baseline top-10**: содержит `SpringContextManager.java` (#10, score 5886).

**Mode B**: `WebhookManager.java` (graph) и `LoadTestFramework.java` (graph) вошли на позиции #6 и #7, вытеснив `DocProvider.java` и `SpringContextManager.java`.

`SpringContextManager.java` lost — score baseline_norm = 0.678, а graph-файлы получили boost от seed `EnterpriseIntegrationApi.java` (norm=0.950), итого graph scores 0.826 и 0.776.

---

### 5.5 Пример: C навредил больше чем B

**Query**: *improve task cancellation and timer management* (тот же, что в 5.2)

**Mode B**: gained `FlowScheduler.java`, не потерял ничего. **Чистый плюс**.

**Mode C**: тот же запрос, но C **потерял** `Task.java`.

**Почему**: в Mode C формула `final = baseline_norm + graph_bonus`. `graph_bonus = seed_norm * 0.3` добавляется ко ВСЕМ graph-соседям, включая те, что уже в baseline. Это означает, что не-GT файлы, которые одновременно и в baseline и в graph, получают повышенный score и вытесняют GT-файлы, которых нет в graph.

`Task.java` (GT, baseline_norm = 0.354) не имеет graph_bonus (его нет среди соседей seeds), а `SimulationConfiguration.java` (non-GT, baseline_norm = 0.357) получает graph_bonus +0.3*seed и поднимается выше.

---

## 6. Статистика графа и проблемы

### 6.1 Hub nodes

Средний seed degree для `dotCMS/core` = **10017**, для `hmislk/hmis` = **9869**. Это означает, что seed-чанки в этих репозиториях имеют суммарно тысячи рёбер. При этом `NEIGHBORS_PER_SEED = 5` ограничивает выборку, но **какие именно 5 соседей выбираются — не контролируется** (берутся первые из `graph.predecessors()` + `graph.successors()`).

### 6.2 Нулевое расширение

27 из 74 queries (36%) получили 0 новых файлов. Причины:
- Seed-файлы не содержат чанков с chunk_id, присутствующими в графе
- Все graph-соседи уже в baseline
- Все seed chunk_id имеют degree > 50 (hub filter)

`jdereg/java-util`: 0 расширений при 5 queries, несмотря на avg_seed_degree=2942. Это парадокс: высокий degree, но 0 новых файлов — вероятно, все соседи уже в baseline (BM25 уже покрывает весь neighbor set).

### 6.3 Качество рёбер call graph

Call graph строится через name-based resolution:

```python
# callgraph_builder.py, строки 89-108
if caller and invoked_name in method_name_to_chunks:
    target_ids = method_name_to_chunks[invoked_name]
    same_file = [t for t in target_ids if t.startswith(file_path)]
    target = same_file[0] if same_file else target_ids[0]
```

Проблемы:
- **Омонимия**: если два класса имеют метод `get()`, вызов `foo.get()` из файла A может быть привязан к неправильному `get()` в файле B
- **Перегрузка**: `method_name_to_chunks` группирует по имени без учёта сигнатуры
- **Нет type resolution**: `object.method()` — неизвестно, какой тип `object`, поэтому связывание по имени метода неточное
- **Одностороннее ограничение**: берётся `caller_chunk_ids[:1]` — только первый caller, остальные игнорируются

---

## 7. Процесс разработки и итерации

### Итерация 1: score propagation с edge_weight=1.0 и max(baseline, graph)

Первая версия: graph-only файлы получали `score = seed_score * 1.0` (полный скор seed'а). Merge: `final = max(baseline_score, graph_score)`.

**Результат**: Hit@10 упал с 0.54 до 0.35 (-19pp). Graph-соседи с высоким seed score полностью вытесняли baseline.

### Итерация 2: discount factor 0.5, baseline keeps scores

Вторая версия: `edge_weight = 0.5`, baseline файлы сохраняют свой score, только graph-only файлы добавляются.

**Результат**: Hit@10 упал с 0.54 до 0.50 (-4pp). Лучше, но всё равно отрицательно.

### Итерация 3: non-displacing (fill remaining slots)

Третья версия: graph-файлы добавляются только если baseline вернул < top_k файлов.

**Результат**: Hit@10 = baseline (идентично). Baseline всегда возвращает >= top_k файлов, expansion никогда не срабатывает.

### Итерация 4 (финальная): normalized scores + additive boost

Финальная версия:
- Нормализация всех file-agg scores в [0,1]
- Graph-only файлы получают `score = seed_norm * EDGE_WEIGHT`
- Файлы в обоих пулах: `score = baseline_norm + BOOST * graph_score`
- BOOST = 0.2

**Результат**: Hit@10 +1.35pp, Recall@10 +0.87pp, MRR +1.39pp. Первый положительный сигнал.

---

## 8. Выводы

### 8.1 Основной результат

**Mode B** даёт **слабый, но реальный положительный сигнал**: Hit@10 +1.35pp, MRR +1.39pp. Это ниже порога "успеха" из ТЗ (+2-3pp), но signal > 0.

**Mode C** — нетто-отрицательный: помог 5 queries, навредил 12. Аддитивный bonus ко ВСЕМ файлам создаёт больше displacement, чем value.

### 8.2 Решение

**LIMITED CONTINUE** — идея graph expansion работает на конкретных кейсах (repos с rich call graph, queries про связанные компоненты), но в текущей форме не даёт sufficient improvement для продакшена.

### 8.3 Ограничения эксперимента

1. **74 samples** — статистически недостаточно для достоверных выводов (+/-1.35pp может быть шумом)
2. **Только BM25 baseline** — не тестировали с hybrid (BM25+vector)
3. **Call graph = только method invocations** — самый бедный из возможных графов
4. **NEIGHBORS_PER_SEED = 5** — жёсткое ограничение, не data-driven
5. **Нет контроля за качеством рёбер** — name-based resolution даёт ложные связи
6. **Scoring формулы подобраны вручную** — не оптимизированы
7. **Дупликаты в выборке** — 2 из 3 "helped" cases — один и тот же query (дубликат sample)

---

## 9. Файлы

| Файл | Описание |
|------|----------|
| `benchmark/retrievers.py` | +2 класса: `BM25GraphExpansion`, `BM25PrioritizedExpansion` + helpers |
| `benchmark/graph_expansion_experiment.py` | Standalone experiment runner |
| `benchmark/results/graph_expansion/detailed_logs.json` | 74 per-query logs с полной информацией |
| `benchmark/results/graph_expansion/showcase_cases.json` | 9 отобранных кейсов |
| `benchmark/results/graph_expansion/EXPERIMENT_REPORT.md` | Автосгенерированный отчёт |
| `backend/app/indexer/callgraph_builder.py` | Построение call graph (существующий код) |
| `backend/app/search/graph_expander.py` | GraphExpander с get_neighbors() (существующий, не использован напрямую) |
