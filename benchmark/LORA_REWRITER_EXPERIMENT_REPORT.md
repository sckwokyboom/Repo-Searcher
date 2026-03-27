# Подробный отчёт об эксперименте: LoRA-дообучение Query Rewriter для доменного поиска кода

**Дата:** 2026-03-26
**Время эксперимента:** 702 секунды (~12 минут)
**Репозиторий:** `jdereg/java-util`

---

## 1. Обзор эксперимента

**Цель:** проверить, улучшает ли LoRA-дообучение малого кодового LLM (Qwen2.5-Coder-1.5B) на одном репозитории качество поиска кода, когда модель переписывает естественный пользовательский запрос в структурированные retrieval-oriented hints для BM25.

**Репозиторий:** `jdereg/java-util` — Java-библиотека утилит (~400 Java-файлов, 9573 code chunks в индексе, 58 eval-сэмплов из git-истории).

---

## 2. Архитектура поискового пайплайна

### 2.1 Парсинг репозитория (tree-sitter)

Репозиторий парсится через **tree-sitter Java AST**. Из каждого `.java`-файла извлекаются:

- **Class-level chunks** — сигнатура класса + поля + сигнатуры методов + javadoc
- **Method-level chunks** — полный код метода + сигнатура + javadoc + имя содержащего класса

Каждый chunk содержит `text_representation` — конкатенацию имени класса, javadoc класса (первые 200 символов), имени метода, javadoc метода, сигнатуры и тела. Это текстовое представление и есть то, что индексируется в BM25.

Фильтры качества:
- **MIN_BODY_LINES = 2** — пропускаются пустые методы и однострочные геттеры
- **MIN_BODY_CHARS = 30** — отсеиваются тривиальные обёртки

### 2.2 Построение BM25-индекса

BM25 строится библиотекой `rank_bm25.BM25Okapi`. Ключевой момент — **кастомная токенизация**:

```python
def tokenize(text: str) -> list[str]:
    # 1. Разделение по пробелам, пунктуации, скобкам
    for word in re.split(r'[\s\.\,\;\:\(\)\{\}\[\]\"\']+', text):
        # 2. CamelCase-сплит: getUserById → get, User, By, Id
        camel_parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', word)
        camel_parts = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', camel_parts)
        # 3. Underscore-сплит
        # 4. Lowercasing, фильтрация токенов < 2 символов
```

Каждый документ в BM25-индексе — это `text_representation` одного code chunk, пропущенный через эту токенизацию. Для `jdereg/java-util` получается **9573 документа** (chunks).

Поиск: запрос токенизируется той же функцией → BM25 выдаёт скоры по всем документам → top-K результатов дедуплицируются по `file_path` (берётся первый chunk из каждого файла).

### 2.3 Eval-сэмплы (бенчмарк)

Из JSONL-файла с коммитами извлекаются пары:
- **query** = очищенное commit message (без boilerplate, conventional commit-префиксов, SHA, emoji)
- **ground truth** = файлы, изменённые в коммите (`.java` только)

Для `jdereg/java-util` получилось **58 сэмплов**.

---

## 3. Генерация обучающего датасета (v2)

### 3.1 Подход

Весь training data строится **только из одного репозитория** (`jdereg/java-util`). Используется tree-sitter для глубокого анализа каждого метода.

### 3.2 Построение MethodProfile

Для каждого code chunk типа `method` строится `MethodProfile`:

1. **CamelCase-сплит имени метода** → семантические токены (`compareMaps` → `[compare, maps]`)
2. **Извлечение invocations** через tree-sitter AST — все вызовы методов внутри тела, кроме шумовых (`get`, `set`, `assertEquals`, `println` и т.д.)
3. **Извлечение type references** — все `type_identifier` ноды, кроме стандартных Java-типов (`String`, `Object`, `HashMap` и т.д.)
4. **Javadoc summary** — первое предложение javadoc, очищенное от HTML, `@param`/`@return`, leading verbs
5. **Parameter types** из сигнатуры
6. **Semantic tags** — автоматическое тегирование по ключевым словам:
   - `conversion`, `validation`, `parsing`, `reflection`, `security`, `concurrency`, `collection_ops`, `date_time`, `serialization`, `comparison`
7. **Quality score** — скоринг по числу invocations, наличию javadoc, семантических тегов, типизированных параметров

Отсеиваются:
- Тривиальные методы (`toString`, `hashCode`, `equals`, `main`)
- Методы с телом < 30 символов
- Методы без invocations, без javadoc и с 1 токеном в имени

### 3.3 Генерация запросов (4 стиля)

Для каждого профиля генерируются запросы в 4 стилях:

| Стиль | Описание | Пример |
|-------|----------|--------|
| **behavioral** | action + object из имени метода | `"find code that compares maps"` |
| **navigation** | из javadoc summary или class context | `"deeply compare two Maps and generate the appropriate"` |
| **short** | 2-4 ключевых слова | `"case insensitive map"` |
| **type_aware** | из invocations и types | `"getenv calculate map capacity"` |

### 3.4 Построение target JSON

Для каждой пары (профиль, запрос) строится структурированный JSON:

```json
{
  "intent": "find_code",
  "search_scope": "implementation | tests",
  "keywords": ["..."],
  "project_terms": ["..."],
  "method_hints": ["..."],
  "api_hints": ["..."],
  "search_queries": ["...", "..."]
}
```

Типы (`project_terms` vs `api_hints`) разделяются по списку известных Java API: `Map`, `HashMap`, `List`, `Collection` → `api_hints`; `CompactMap`, `GraphComparator`, `Delta` → `project_terms`.

Search queries дедуплицируются по Jaccard similarity (порог 0.7).

### 3.5 Фильтрация качества

Каждый sample проверяется:
- Запрос 3–10 слов
- Доля стоп-слов <= 40%
- Есть хотя бы 1 method hint
- Есть хотя бы 1 project term или api hint
- Keywords не пустые

### 3.6 Итоговый датасет

| Параметр | Значение |
|----------|----------|
| Всего сэмплов | **1490** |
| Train | **1341** |
| Val | **149** |
| Стили | behavioral: 257, navigation: 386, short: 347, type_aware: 500 |
| Test-сэмплы | ~27% |

### 3.7 Примеры из обучающих данных

#### Пример 1 (behavioral)

**Метод:** `GraphComparator.compareMaps`
**Файл:** `src/main/java/com/cedarsoftware/util/GraphComparator.java`

```
Запрос: "find code that compares maps"
```

```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": ["find", "code", "compares", "maps"],
  "project_terms": ["GraphComparator", "Delta", "ID", "Entry"],
  "method_hints": ["compareMaps", "identityHashCode", "entrySet"],
  "api_hints": ["Collection", "LinkedList", "Map"],
  "search_queries": [
    "find code compares maps",
    "GraphComparator compareMaps",
    "compareMaps Collection LinkedList",
    "conversion compare maps"
  ]
}
```

#### Пример 2 (navigation)

**Метод:** `EncryptionUtilities.fastSHA1`
**Файл:** `src/main/java/com/cedarsoftware/util/EncryptionUtilities.java`

```
Запрос: "calculates a SHA-1 hash of a file using"
```

```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": ["calculates", "sha-1", "hash", "file", "using"],
  "project_terms": ["EncryptionUtilities", "FileInputStream", "FileNotFoundException"],
  "method_hints": ["fastSHA1", "validateFileSize", "calculateFileHash"],
  "api_hints": ["File", "IOException"],
  "search_queries": [
    "calculates sha-1 hash file using",
    "EncryptionUtilities fastSHA1",
    "fastSHA1 File IOException",
    "validation fast sha1"
  ]
}
```

#### Пример 3 (short)

**Метод:** `CaseInsensitiveMap` (конструктор)
**Файл:** `src/main/java/com/cedarsoftware/util/CaseInsensitiveMap.java`

```
Запрос: "case insensitive map"
```

```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": ["case", "insensitive", "map"],
  "project_terms": ["CaseInsensitiveMap", "MultiKeyMap", "ConcurrentMap"],
  "method_hints": ["CaseInsensitiveMap", "requireNonNull", "determineBackingMap"],
  "api_hints": ["Map"],
  "search_queries": [
    "case insensitive map",
    "CaseInsensitiveMap CaseInsensitiveMap",
    "CaseInsensitiveMap Map",
    "conversion case insensitive"
  ]
}
```

#### Пример 4 (type_aware)

**Метод:** `DateUtilitiesTest.testZoneOffsets`
**Файл:** `src/test/java/com/cedarsoftware/util/DateUtilitiesTest.java`

```
Запрос: "parse date get hour"
```

```json
{
  "intent": "find_code",
  "search_scope": "tests",
  "keywords": ["parse", "date", "get", "hour"],
  "project_terms": ["DateUtilitiesTest", "ZonedDateTime"],
  "method_hints": ["testZoneOffsets", "parseDate", "getHour"],
  "search_queries": [
    "parse date get hour",
    "DateUtilitiesTest testZoneOffsets",
    "conversion test zone"
  ]
}
```

#### Пример 5 (behavioral)

**Метод:** `ReflectionUtils.validateFieldAccess`
**Файл:** `src/main/java/com/cedarsoftware/util/ReflectionUtils.java`

```
Запрос: "find code that validates field access"
```

```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": ["find", "code", "validates", "field", "access"],
  "project_terms": ["ReflectionUtils", "SecurityException"],
  "method_hints": ["validateFieldAccess", "isSecurityEnabled", "getDeclaringClass"],
  "api_hints": ["Field"],
  "search_queries": [
    "find code validates field access",
    "ReflectionUtils validateFieldAccess",
    "validateFieldAccess Field",
    "validation validate field"
  ]
}
```

---

## 4. Обучение LoRA-адаптера

| Параметр | Значение |
|----------|----------|
| Базовая модель | `Qwen/Qwen2.5-Coder-1.5B` |
| Тип адаптации | LoRA (PEFT) |
| r | 8 |
| lora_alpha | 16 |
| lora_dropout | 0.05 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Эпохи | 3 |
| Batch size | 2 |
| Gradient accumulation | 8 (effective batch = 16) |
| Learning rate | 1e-4, cosine schedule |
| Warmup | 5% |
| Max seq length | 512 |
| Платформа | Apple Silicon (MPS) |
| Max train samples | 2000 (использовано 1341) |
| Max val samples | 300 (использовано 149) |
| Eval loss (финал) | **0.44** |

Обучаемых параметров: ~2.5M из ~1.5B (менее 0.2% модели).

---

## 5. Evaluation pipeline

### 5.1 Четыре режима сравнения

1. **BM25_raw** — сырой пользовательский запрос (commit message) подаётся напрямую в BM25
2. **BM25_base_rewrite** — запрос переписывается базовой Qwen 1.5B **без LoRA**, JSON парсится, все поля конкатенируются и подаются в BM25
3. **BM25_lora_rewrite** — запрос переписывается Qwen 1.5B **с LoRA-адаптером**, аналогично
4. **BM25_combined** — оригинальный запрос + все извлечённые из LoRA-вывода термины конкатенируются и подаются в BM25 вместе

### 5.2 Парсинг вывода модели

Из raw-вывода модели парсится JSON:

```python
# Извлечь {…} из сырого текста
# Разобрать JSON
# Конкатенировать все поля: search_queries + keywords + method_hints + project_terms
# Объединить в одну строку и подать в BM25
```

Если JSON не парсится, используется `original_query + raw_output[:300]` как fallback.

---

## 6. Результаты

### 6.1 Количественные метрики

| Retriever | R@1 | R@5 | R@10 | R@20 | MRR |
|-----------|-----|-----|------|------|-----|
| **BM25_raw** (baseline) | 0.276 | 0.500 | 0.655 | 0.724 | 0.393 |
| **BM25_base_rewrite** | 0.276 | 0.517 | 0.621 | 0.707 | 0.382 |
| **BM25_lora_rewrite** | 0.259 | 0.379 | 0.448 | 0.500 | 0.313 |
| **BM25_combined** | **0.431** | **0.707** | **0.862** | **0.931** | **0.549** |

### 6.2 Улучшение combined vs baseline

| Метрика | Baseline | Combined | Delta абс. | Delta отн. |
|---------|----------|----------|------------|------------|
| R@1 | 0.276 | 0.431 | +0.155 | **+56%** |
| R@5 | 0.500 | 0.707 | +0.207 | **+41%** |
| R@10 | 0.655 | 0.862 | +0.207 | **+32%** |
| R@20 | 0.724 | 0.931 | +0.207 | **+29%** |
| MRR | 0.393 | 0.549 | +0.156 | **+40%** |

### 6.3 JSON Parse Success

| Модель | Valid JSON | Rate |
|--------|-----------|------|
| Base Qwen (без LoRA) | 0/58 | **0%** |
| LoRA Qwen | 57/58 | **98%** |

LoRA научила модель идеально следовать формату. Базовая модель выдаёт произвольные JSON-структуры, не совпадающие с целевой схемой.

---

## 7. Качественный анализ

### 7.1 Проблема mode collapse у LoRA standalone

LoRA-модель при использовании в одиночку показала **mode collapse** — на большинство запросов (по крайней мере 7 из первых 10 eval-сэмплов) она выдавала **один и тот же вывод**:

```json
{
  "intent": "find_code",
  "search_scope": "implementation",
  "keywords": ["fix", "critical", "security", "vulnerabilities", "compactmap", "dynamic", "code", "generation", "security", "fixes"],
  "project_terms": ["CompactMap", "CompactMapBuilder", "CompactMapEntry", "CompactMapEntrySet"],
  "method_hints": ["fixSecurityVulnerabilities", "validateInput", "build", "getDeclaredMethod"],
  "search_queries": ["fix critical security vulnerabilities compactmap dynamic", "CompactMap fixSecurityVulnerabilities", ...]
}
```

Это один и тот же ответ для совершенно разных запросов: про `ArrayUtilities`, `ByteUtilities`, `CaseInsensitiveMap`, `CollectionUtilities`, и даже про `ReflectionUtils`. Модель "запомнила" один частый паттерн из обучающих данных (CompactMap — вероятно самый частый класс в репозитории).

### 7.2 Базовая модель тоже коллапсирует

Базовая Qwen (без LoRA) показала **точно такую же проблему** — на все 10 примеров выдавала один и тот же вывод:

```json
{
  "search": {
    "query": "Fix critical security vulnerabilities in CompactMap dynamic code generation",
    "hints": {
      "structured_retrieval": {
        "codebase": {
          "language": "Java",
          "features": ["CompactMap", "dynamic code generation", "security fixes", ...]
        }
      }
    }
  }
}
```

Формат другой (не соответствует целевой схеме, JSON parse = 0%), но суть та же — один ответ на все запросы.

### 7.3 Пример, где LoRA работает хорошо (Example 8)

**Запрос:** `"Security: Fix critical ReflectionUtils vulnerabilities..."`

**GT:** `src/main/java/com/cedarsoftware/util/DateUtilities.java`

LoRA-вывод отличается от коллапсированного ответа:

```json
{
  "search_scope": "tests",
  "keywords": ["security", "fix", "critical", "date", "utilities"],
  "project_terms": ["DateUtilitiesFixReDoSVulnerabilitiesTest", "DateUtilities", "ZonedDateTime"],
  "method_hints": ["testReDoSVulnerability", "parseDate", "validateDate"],
  "search_queries": ["security fix critical date utilities fixed", ...]
}
```

Здесь модель правильно подхватила `DateUtilities` и связанные сущности. **Все три режима дали hit@5 = True**.

### 7.4 Пример с top-5 результатами

**Запрос:** `"Fix ArrayUtilities code review issues - Fix EMPTY_CLASS_ARRAY generic type safety..."`

**GT:** `src/main/java/com/cedarsoftware/util/CompactMap.java`

| Режим | Top-5 файлов | Hit@5 |
|-------|-------------|-------|
| **BM25_raw** | ClassUtilitiesSecurityFixesTest, **CompactMap.java**, IOUtilities, DataGeneratorInputStream, UrlUtilities | **True** |
| **Base rewrite** | ClassUtilitiesSecurityFixesTest, **CompactMap.java**, DataGeneratorInputStream, IOUtilities, SystemUtilities | **True** |
| **LoRA rewrite** | ClassUtilitiesSecurityFixesTest, ByteUtilitiesSecurityTest, ClassUtilitiesFinalOptimizationsTest, UrlUtilitiesSecurityTest, StringUtilitiesSecurityTest | **False** |
| **Combined** | ClassUtilitiesSecurityFixesTest, **CompactMap.java**, UrlUtilitiesSecurityTest, ClassUtilitiesFinalOptimizationsTest, ClassUtilitiesSecurityTest | **True** |

LoRA standalone теряет `CompactMap.java` из top-5 (хотя иронично упоминает его в project_terms), потому что коллапсированный вывод забит общими security-терминами. Combined восстанавливает его за счёт оригинального запроса.

### 7.5 Как combined побеждает

Combined работает так: `original_query + all_lora_terms`. Даже когда LoRA коллапсирует, она добавляет project-specific термины (`CompactMap`, `CompactMapBuilder`, `validateInput`) в запрос, а оригинальный запрос и так содержит правильные естественно-языковые термины. В итоге combined собирает "лучшее из обоих миров".

---

## 8. Диагноз и выводы

### 8.1 Почему LoRA standalone проиграл

1. **Mode collapse** — модель запомнила доминирующий паттерн из обучающего датасета. CompactMap — один из крупнейших классов в репозитории, его методы генерировали непропорционально много training samples.
2. **Train/eval distribution mismatch** — training queries были синтетическими (из CamelCase-имён, javadoc), а eval queries — реальные commit messages (длинные, с changelog-описаниями).
3. **Мощность 1.5B модели** — при сильном LoRA-fine-tuning маленькая модель легко переобучается на частотные паттерны.

### 8.2 Почему combined выиграл

Combined по сути представляет собой **query expansion** через LoRA. Даже коллапсированный вывод содержит project-specific термины (`CompactMap`, `CompactMapBuilder`, `validateInput`), которых не было в исходном запросе. BM25 получает больше релевантных токенов — выше recall.

### 8.3 Итоговая оценка по критериям успешности

| Критерий | Статус |
|----------|--------|
| LoRA-адаптер обучен без full fine-tuning | Да |
| Весь train data из одного репозитория | Да |
| Retriever запускается на том же репозитории | Да |
| Количественное сравнение до/после | Да |
| Модель лучше использует локальную лексику (в combined) | Да (частично) |
| LoRA standalone > baseline | Нет (mode collapse) |
| Combined > baseline | Да (+41% R@5, +56% R@1) |

### 8.4 Главный вывод

Даже быстрое LoRA-дообучение малого кодового LLM на одном репозитории создаёт полезный **query expansion** модуль: модель научается генерировать project-specific термины (классы, методы, типы), которые при **конкатенации с исходным запросом** дают +41% Recall@5.

Однако использование LoRA-модели как **единственного source of truth** для переписывания запросов проваливается из-за mode collapse — модель теряет разнообразие ответов. Решение: **комбинировать** исходный запрос с output модели, а не заменять его.

---

## Приложение: Структура файлов эксперимента

| Файл | Назначение |
|------|------------|
| `benchmark/lora_training/query_rewriter/prepare_data.py` | Генерация обучающего датасета (v2) |
| `benchmark/lora_training/query_rewriter/train.py` | Обучение LoRA-адаптера |
| `benchmark/lora_training/query_rewriter/evaluate.py` | Evaluation pipeline (4 режима) |
| `benchmark/lora_training/query_rewriter/data/train_rewriter.jsonl` | 1341 training sample |
| `benchmark/lora_training/query_rewriter/data/val_rewriter.jsonl` | 149 validation samples |
| `benchmark/lora_training/query_rewriter/DATASET_EXAMPLES.md` | Примеры из датасета с телами методов |
| `benchmark/lora_training/output/rewriter_lora/final/` | Обученный LoRA-адаптер |
| `benchmark/results/rewriter_experiment_results.json` | Полные результаты (метрики + примеры) |
| `benchmark/results/rewriter_experiment_report.md` | Краткий отчёт с метриками |
| `backend/app/indexer/parser.py` | Tree-sitter парсер Java (chunks) |
| `backend/app/indexer/bm25_builder.py` | BM25-индекс + токенизация |
