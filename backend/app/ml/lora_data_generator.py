"""
Generate LoRA training data from indexed CodeChunks.

Adapted from benchmark/lora_training/query_rewriter/prepare_data.py
for online use within the backend.
"""

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import tree_sitter
import tree_sitter_java as tsjava

from app.indexer.bm25_builder import tokenize
from app.models.search import CodeChunk

SYSTEM_PROMPT = (
    "Rewrite this search query into structured retrieval hints "
    "for a Java codebase.\n\nQuery: {query}\n\nOutput JSON:"
)

NOISE_INVOCATIONS = {
    "get", "set", "put", "add", "remove", "size", "isEmpty", "contains",
    "equals", "hashCode", "toString", "valueOf", "iterator", "next", "hasNext",
    "assertEquals", "assertTrue", "assertFalse", "assertNotNull", "assertNull",
    "assertThrows", "assertSame", "assertNotSame", "fail",
    "before", "after", "beforeEach", "afterEach",
    "println", "print", "format", "append", "length", "substring",
    "getClass", "getName", "getMessage", "getKey", "getValue",
    "close", "flush", "write", "read",
}

NOISE_TYPES = {
    "String", "Object", "Integer", "Long", "Boolean", "Double", "Float",
    "Byte", "Short", "Character", "Void", "Class", "Number",
    "Override", "Deprecated", "SuppressWarnings", "Test", "BeforeEach",
    "AfterEach", "DisplayName", "ParameterizedTest",
}

JAVA_API_TYPES = {
    "Map", "HashMap", "TreeMap", "LinkedHashMap", "ConcurrentHashMap",
    "List", "ArrayList", "LinkedList", "Set", "HashSet", "TreeSet",
    "Collection", "Collections", "Arrays", "Iterator", "Iterable",
    "Comparator", "Comparable", "Optional",
    "BitSet", "BigDecimal", "BigInteger",
    "Date", "Calendar", "LocalDate", "LocalDateTime", "Instant", "Duration",
    "Pattern", "Matcher", "StringBuilder", "StringBuffer",
    "File", "Path", "InputStream", "OutputStream", "Reader", "Writer",
    "URL", "URI", "Socket", "ServerSocket",
    "Thread", "Runnable", "Callable", "Future", "ExecutorService",
    "AtomicInteger", "AtomicLong", "AtomicReference", "AtomicBoolean",
    "Lock", "ReentrantLock", "CountDownLatch", "Semaphore",
    "Field", "Method", "Constructor", "Modifier",
    "ClassLoader", "SecurityManager",
    "Exception", "RuntimeException", "IOException", "IllegalArgumentException",
    "NullPointerException", "IndexOutOfBoundsException",
    "JsonObject", "JsonArray", "JsonElement",
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "for", "of", "in", "on", "at", "to", "from", "by", "with", "as",
    "and", "or", "but", "not", "no", "nor", "so", "yet",
    "this", "that", "these", "those", "it", "its",
    "where", "which", "what", "who", "whom", "whose", "when", "how", "why",
    "if", "then", "else", "than", "also", "just", "only", "very",
    "all", "each", "every", "some", "any", "few", "more", "most",
    "here", "there", "up", "down", "out", "into",
}

TRIVIAL_METHODS = {
    "toString", "hashCode", "equals", "main", "setUp", "tearDown",
    "init", "destroy", "clone", "finalize", "compareTo",
}

TAG_RULES = {
    "conversion": {"convert", "transform", "parse", "from", "Converter", "cast", "coerce", "map"},
    "validation": {"validate", "check", "verify", "require", "ensure", "guard", "constraint"},
    "parsing": {"parse", "read", "decode", "deserialize", "unmarshal", "extract", "scan"},
    "reflection": {"reflect", "invoke", "getMethod", "getField", "getDeclaredField",
                   "setAccessible", "newInstance", "forName"},
    "security": {"security", "sanitize", "encrypt", "decrypt", "hash", "token", "credential",
                 "permission", "access", "authorize"},
    "concurrency": {"synchronized", "atomic", "concurrent", "lock", "thread", "semaphore",
                    "latch", "barrier", "volatile"},
    "collection_ops": {"sort", "filter", "merge", "flatten", "group", "partition", "collect",
                       "stream", "reduce", "aggregate"},
    "date_time": {"date", "time", "calendar", "temporal", "instant", "duration", "period",
                  "zone", "epoch", "timestamp"},
    "serialization": {"serialize", "json", "xml", "marshal", "toJson", "fromJson", "toXml"},
    "comparison": {"compare", "diff", "deep", "shallow", "Comparator", "ordering"},
}

# Tree-sitter singleton
_ts_parser = None


def _get_parser():
    global _ts_parser
    if _ts_parser is None:
        lang = tree_sitter.Language(tsjava.language())
        _ts_parser = tree_sitter.Parser(lang)
    return _ts_parser


def _extract_invocations_and_types(body: str) -> tuple[list[str], list[str]]:
    parser = _get_parser()
    wrapped = f"class X {{ {body} }}"
    tree = parser.parse(wrapped.encode("utf-8"))

    invocations = []
    types = []

    def walk(node):
        if node.type == "method_invocation":
            method_name = None
            for child in node.children:
                if child.type == "identifier":
                    method_name = child.text.decode()
            if method_name and method_name not in NOISE_INVOCATIONS and len(method_name) > 2:
                invocations.append(method_name)
        elif node.type == "type_identifier":
            type_name = node.text.decode()
            if type_name not in NOISE_TYPES and len(type_name) > 1:
                types.append(type_name)
        for child in node.children:
            walk(child)

    walk(tree.root_node)

    seen_inv = set()
    unique_inv = []
    for inv in invocations:
        if inv not in seen_inv:
            seen_inv.add(inv)
            unique_inv.append(inv)

    seen_t = set()
    unique_types = []
    for t in types:
        if t not in seen_t:
            seen_t.add(t)
            unique_types.append(t)

    return unique_inv[:10], unique_types[:10]


def _split_camel_case(name: str) -> list[str]:
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", parts)
    return [p.lower() for p in parts.split() if len(p) > 1]


def _extract_javadoc_summary(javadoc: str | None) -> str | None:
    if not javadoc:
        return None
    doc = javadoc.strip()
    doc = re.sub(r"/\*\*\s*", "", doc)
    doc = re.sub(r"\*/\s*$", "", doc)
    doc = re.sub(r"\n\s*\*\s?", "\n", doc)
    doc = re.sub(r"^\s*\*\s*", "", doc)
    doc = doc.strip()
    doc = re.sub(r"@\w+.*", "", doc, flags=re.DOTALL).strip()
    if not doc:
        return None
    first_sent = re.split(r"[.\n]", doc)[0].strip()
    first_sent = re.sub(r"<[^>]+>", "", first_sent).strip()
    if len(first_sent) < 10 or len(first_sent) > 200:
        return None
    first_sent = re.sub(
        r"^(Returns?|Gets?|Sets?|Creates?|Builds?|Converts?|Checks?|Finds?|Determines?|Provides?|Computes?)\s+",
        "", first_sent,
    )
    first_sent = first_sent.strip()
    if len(first_sent) < 8:
        return None
    return first_sent[0].lower() + first_sent[1:]


def _extract_param_types(signature: str) -> list[str]:
    m = re.search(r"\(([^)]*)\)", signature)
    if not m:
        return []
    params = m.group(1)
    types = re.findall(r"\b([A-Z][a-zA-Z0-9]+)\b", params)
    return [t for t in types if t not in NOISE_TYPES and len(t) > 2]


def _extract_return_type(signature: str) -> str | None:
    m = re.match(r".*?\b([A-Z][a-zA-Z0-9<>,\s]+?)\s+\w+\s*\(", signature)
    if m:
        ret = m.group(1).strip()
        ret = re.sub(r"<.*>", "", ret).strip()
        if ret not in NOISE_TYPES and ret not in {"public", "private", "protected", "static", "final", "abstract"}:
            return ret
    return None


def _infer_semantic_tags(method_tokens: list[str], invocations: list[str], types: list[str]) -> list[str]:
    all_terms = set()
    for t in method_tokens:
        all_terms.add(t.lower())
    for inv in invocations:
        for part in _split_camel_case(inv):
            all_terms.add(part.lower())
    for typ in types:
        all_terms.add(typ)
        for part in _split_camel_case(typ):
            all_terms.add(part.lower())

    tags = []
    for tag, keywords in TAG_RULES.items():
        if all_terms & {k.lower() for k in keywords}:
            tags.append(tag)
    return tags[:3]


def _classify_type(type_name: str) -> str:
    if type_name in JAVA_API_TYPES:
        return "api"
    return "project"


# --- Method Profile ---

@dataclass
class MethodProfile:
    chunk: CodeChunk
    class_name: str
    method_name: str
    file_path: str
    is_test: bool
    method_tokens: list[str]
    class_tokens: list[str]
    javadoc_summary: str | None
    param_types: list[str]
    return_type: str | None
    top_invocations: list[str]
    top_types: list[str]
    semantic_tags: list[str]
    quality_score: float = 0.0


def build_profile(chunk: CodeChunk) -> MethodProfile | None:
    if chunk.chunk_type != "method" or not chunk.method_name:
        return None
    if chunk.method_name in TRIVIAL_METHODS:
        return None
    if not chunk.body or len(chunk.body) < 30:
        return None

    method_tokens = _split_camel_case(chunk.method_name)
    if len(method_tokens) < 1:
        return None

    class_tokens = _split_camel_case(chunk.class_name) if chunk.class_name else []
    invocations, types = _extract_invocations_and_types(chunk.body)

    if len(invocations) < 1 and not chunk.javadoc and len(method_tokens) < 2:
        return None

    javadoc_summary = _extract_javadoc_summary(chunk.javadoc)
    param_types = _extract_param_types(chunk.signature) if chunk.signature else []
    return_type = _extract_return_type(chunk.signature) if chunk.signature else None
    is_test = "src/test" in chunk.file_path

    semantic_tags = _infer_semantic_tags(method_tokens, invocations, types)

    score = 0.0
    score += min(len(invocations), 5) * 0.5
    score += (1.0 if javadoc_summary else 0.0) * 2.0
    score += min(len(semantic_tags), 2) * 1.0
    score += min(len(param_types), 3) * 0.3
    score -= (1.0 if is_test else 0.0) * 0.5

    return MethodProfile(
        chunk=chunk,
        class_name=chunk.class_name or "",
        method_name=chunk.method_name,
        file_path=chunk.file_path,
        is_test=is_test,
        method_tokens=method_tokens,
        class_tokens=class_tokens,
        javadoc_summary=javadoc_summary,
        param_types=param_types,
        return_type=return_type,
        top_invocations=invocations,
        top_types=types,
        semantic_tags=semantic_tags,
        quality_score=score,
    )


# --- Query generators ---

def _query_behavioral(profile: MethodProfile) -> str | None:
    if len(profile.method_tokens) < 2:
        return None
    action = profile.method_tokens[0]
    obj = " ".join(profile.method_tokens[1:4])

    templates = [
        f"where is {obj} {action}d",
        f"which code handles {action} {obj}",
        f"how does the code {action} {obj}",
        f"find code that {action}s {obj}",
    ]
    if profile.semantic_tags:
        tag = profile.semantic_tags[0].replace("_", " ")
        templates.append(f"where is {tag} for {obj}")

    query = random.choice(templates)
    words = query.split()
    if len(words) < 3 or len(words) > 10:
        return None
    return query


def _query_navigation(profile: MethodProfile) -> str | None:
    if profile.javadoc_summary:
        words = profile.javadoc_summary.split()[:8]
        if len(words) >= 3:
            return " ".join(words)

    if profile.class_name and profile.semantic_tags:
        tag = profile.semantic_tags[0].replace("_", " ")
        return f"{tag} in {profile.class_name}"

    if profile.class_name and len(profile.method_tokens) >= 2:
        concept = " ".join(profile.method_tokens[:3])
        return f"find {concept} in {profile.class_name}"

    return None


def _query_short(profile: MethodProfile) -> str | None:
    candidates = []
    for t in profile.method_tokens:
        if t.lower() not in STOPWORDS and len(t) > 2:
            candidates.append(t)
    for inv in profile.top_invocations[:2]:
        for part in _split_camel_case(inv):
            if part.lower() not in STOPWORDS and len(part) > 2:
                candidates.append(part)
    for tag in profile.semantic_tags[:1]:
        candidates.append(tag.replace("_", " "))

    seen = set()
    unique = []
    for c in candidates:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            unique.append(c)

    if len(unique) < 2:
        return None

    count = min(random.randint(2, 4), len(unique))
    return " ".join(unique[:count])


def _query_type_aware(profile: MethodProfile) -> str | None:
    parts = []
    for inv in profile.top_invocations[:2]:
        parts.extend(_split_camel_case(inv))
    for t in profile.top_types[:2]:
        if t not in NOISE_TYPES:
            parts.extend(_split_camel_case(t))

    seen = set()
    unique = []
    for p in parts:
        pl = p.lower()
        if pl not in seen and pl not in STOPWORDS and len(pl) > 2:
            seen.add(pl)
            unique.append(p)

    if len(unique) < 2:
        return None
    return " ".join(unique[:4])


def _build_target(profile: MethodProfile, query: str) -> dict:
    raw_keywords = tokenize(query)
    keywords = [k for k in raw_keywords if k.lower() not in STOPWORDS]
    keywords = list(dict.fromkeys(keywords))[:6]

    project_terms = []
    api_hints = []

    if profile.class_name:
        if _classify_type(profile.class_name) == "api":
            api_hints.append(profile.class_name)
        else:
            project_terms.append(profile.class_name)

    for t in profile.top_types:
        bucket = api_hints if _classify_type(t) == "api" else project_terms
        if t not in bucket:
            bucket.append(t)

    for pt in profile.param_types:
        bucket = api_hints if _classify_type(pt) == "api" else project_terms
        if pt not in bucket:
            bucket.append(pt)

    project_terms = project_terms[:4]
    api_hints = api_hints[:4]

    method_hints = [profile.method_name]
    for inv in profile.top_invocations[:2]:
        if inv != profile.method_name:
            method_hints.append(inv)
    method_hints = method_hints[:3]

    search_queries = []
    if keywords:
        search_queries.append(" ".join(keywords[:5]))
    if profile.class_name and profile.method_name:
        search_queries.append(f"{profile.class_name} {profile.method_name}")
    type_parts = []
    if profile.method_name:
        type_parts.append(profile.method_name)
    type_parts.extend(api_hints[:2])
    if len(type_parts) >= 2:
        search_queries.append(" ".join(type_parts))
    if profile.semantic_tags and profile.method_tokens:
        tag = profile.semantic_tags[0].replace("_", " ")
        action = " ".join(profile.method_tokens[:2])
        search_queries.append(f"{tag} {action}")

    search_queries = _deduplicate_queries(search_queries)

    return {
        "intent": "find_code",
        "search_scope": "tests" if profile.is_test else "implementation",
        "keywords": keywords,
        "project_terms": project_terms,
        "method_hints": method_hints,
        "api_hints": api_hints,
        "search_queries": search_queries,
    }


def _deduplicate_queries(queries: list[str]) -> list[str]:
    if len(queries) <= 1:
        return queries
    result = [queries[0]]
    for q in queries[1:]:
        q_tokens = set(q.lower().split())
        is_dup = False
        for existing in result:
            e_tokens = set(existing.lower().split())
            if not q_tokens or not e_tokens:
                continue
            jaccard = len(q_tokens & e_tokens) / len(q_tokens | e_tokens)
            if jaccard > 0.7:
                is_dup = True
                break
        if not is_dup:
            result.append(q)
    return result[:4]


def _is_quality_sample(query: str, target: dict) -> bool:
    words = query.split()
    if len(words) < 3 or len(words) > 10:
        return False
    sw_count = sum(1 for w in words if w.lower() in STOPWORDS)
    if sw_count / len(words) > 0.4:
        return False
    if not target.get("method_hints"):
        return False
    if not target.get("project_terms") and not target.get("api_hints"):
        return False
    if not target.get("keywords"):
        return False
    return True


# --- Public API ---

def generate_training_data(chunks: list[CodeChunk], seed: int = 42) -> tuple[list[dict], list[dict], int]:
    """Generate LoRA training data from indexed chunks.

    Returns (train_samples, val_samples, num_profiles).
    Each sample is a dict with 'prompt' and 'completion' keys.
    """
    random.seed(seed)

    # Build profiles
    profiles = []
    for chunk in chunks:
        profile = build_profile(chunk)
        if profile is not None:
            profiles.append(profile)

    if not profiles:
        return [], [], 0

    # Select quality methods
    profiles.sort(key=lambda p: p.quality_score, reverse=True)
    max_test = int(500 * 0.25)
    max_impl = 500 - max_test
    selected_impl = [p for p in profiles if not p.is_test][:max_impl]
    selected_test = [p for p in profiles if p.is_test][:max_test]
    selected = selected_impl + selected_test
    random.shuffle(selected)

    # Generate samples
    generators = [
        ("behavioral", _query_behavioral),
        ("navigation", _query_navigation),
        ("short", _query_short),
        ("type_aware", _query_type_aware),
    ]

    samples = []
    for profile in selected:
        for _, gen_fn in generators:
            query = gen_fn(profile)
            if query is None:
                continue
            target = _build_target(profile, query)
            if not _is_quality_sample(query, target):
                continue
            prompt = SYSTEM_PROMPT.format(query=query)
            completion = json.dumps(target, ensure_ascii=False)
            samples.append({"prompt": prompt, "completion": completion})

    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train = samples[:split]
    val = samples[split:]

    return train, val, len(profiles)


def fast_estimate_samples(chunk_count: int) -> int:
    """Fast sample count estimate without generating data.

    ~60% of chunks produce profiles, each yields ~2 usable samples.
    """
    estimated_profiles = int(chunk_count * 0.6)
    estimated_samples = int(estimated_profiles * 2 * 0.9)  # 90% train split
    return min(estimated_samples, 2000)  # capped at 2000


def estimate_training_time(
    num_samples: int,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 8,
) -> float:
    """Estimate training time in minutes.

    Based on ~0.9s per step on Apple Silicon MPS (M-series).
    """
    steps_per_epoch = max(1, num_samples // (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    seconds = total_steps * 0.9
    return round(seconds / 60, 1)
