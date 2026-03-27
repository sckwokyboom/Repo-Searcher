import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter
import tree_sitter_java as tsjava

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "backend"))

from backend.app.indexer.bm25_builder import tokenize
from backend.app.indexer.store import load_chunks
from backend.app.models.search import CodeChunk

REPO_ID = "jdereg--java-util"
DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = (
    "Rewrite this search query into structured retrieval hints "
    "for a Java codebase.\n\nQuery: {query}\n\nOutput JSON:"
)

NOISE_INVOCATIONS = {
    "get",
    "set",
    "put",
    "add",
    "remove",
    "size",
    "isEmpty",
    "contains",
    "equals",
    "hashCode",
    "toString",
    "valueOf",
    "iterator",
    "next",
    "hasNext",
    "assertEquals",
    "assertTrue",
    "assertFalse",
    "assertNotNull",
    "assertNull",
    "assertThrows",
    "assertSame",
    "assertNotSame",
    "fail",
    "before",
    "after",
    "beforeEach",
    "afterEach",
    "println",
    "print",
    "format",
    "append",
    "length",
    "substring",
    "getClass",
    "getName",
    "getMessage",
    "getKey",
    "getValue",
    "close",
    "flush",
    "write",
    "read",
}

NOISE_TYPES = {
    "String",
    "Object",
    "Integer",
    "Long",
    "Boolean",
    "Double",
    "Float",
    "Byte",
    "Short",
    "Character",
    "Void",
    "Class",
    "Number",
    "Override",
    "Deprecated",
    "SuppressWarnings",
    "Test",
    "BeforeEach",
    "AfterEach",
    "DisplayName",
    "ParameterizedTest",
}

JAVA_API_TYPES = {
    "Map",
    "HashMap",
    "TreeMap",
    "LinkedHashMap",
    "ConcurrentHashMap",
    "List",
    "ArrayList",
    "LinkedList",
    "Set",
    "HashSet",
    "TreeSet",
    "Collection",
    "Collections",
    "Arrays",
    "Iterator",
    "Iterable",
    "Comparator",
    "Comparable",
    "Optional",
    "BitSet",
    "BigDecimal",
    "BigInteger",
    "Date",
    "Calendar",
    "LocalDate",
    "LocalDateTime",
    "Instant",
    "Duration",
    "Pattern",
    "Matcher",
    "StringBuilder",
    "StringBuffer",
    "File",
    "Path",
    "InputStream",
    "OutputStream",
    "Reader",
    "Writer",
    "URL",
    "URI",
    "Socket",
    "ServerSocket",
    "Thread",
    "Runnable",
    "Callable",
    "Future",
    "ExecutorService",
    "AtomicInteger",
    "AtomicLong",
    "AtomicReference",
    "AtomicBoolean",
    "Lock",
    "ReentrantLock",
    "CountDownLatch",
    "Semaphore",
    "Field",
    "Method",
    "Constructor",
    "Modifier",
    "ClassLoader",
    "SecurityManager",
    "Exception",
    "RuntimeException",
    "IOException",
    "IllegalArgumentException",
    "NullPointerException",
    "IndexOutOfBoundsException",
    "JsonObject",
    "JsonArray",
    "JsonElement",
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "need",
    "must",
    "for",
    "of",
    "in",
    "on",
    "at",
    "to",
    "from",
    "by",
    "with",
    "as",
    "and",
    "or",
    "but",
    "not",
    "no",
    "nor",
    "so",
    "yet",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "where",
    "which",
    "what",
    "who",
    "whom",
    "whose",
    "when",
    "how",
    "why",
    "if",
    "then",
    "else",
    "than",
    "also",
    "just",
    "only",
    "very",
    "all",
    "each",
    "every",
    "some",
    "any",
    "few",
    "more",
    "most",
    "here",
    "there",
    "up",
    "down",
    "out",
    "into",
}

TRIVIAL_METHODS = {
    "toString",
    "hashCode",
    "equals",
    "main",
    "setUp",
    "tearDown",
    "init",
    "destroy",
    "clone",
    "finalize",
    "compareTo",
}

TAG_RULES = {
    "conversion": {
        "convert",
        "transform",
        "parse",
        "from",
        "Converter",
        "cast",
        "coerce",
        "map",
    },
    "validation": {
        "validate",
        "check",
        "verify",
        "require",
        "ensure",
        "guard",
        "constraint",
    },
    "parsing": {
        "parse",
        "read",
        "decode",
        "deserialize",
        "unmarshal",
        "extract",
        "scan",
    },
    "reflection": {
        "reflect",
        "invoke",
        "getMethod",
        "getField",
        "getDeclaredField",
        "setAccessible",
        "newInstance",
        "forName",
    },
    "security": {
        "security",
        "sanitize",
        "encrypt",
        "decrypt",
        "hash",
        "token",
        "credential",
        "permission",
        "access",
        "authorize",
    },
    "concurrency": {
        "synchronized",
        "atomic",
        "concurrent",
        "lock",
        "thread",
        "semaphore",
        "latch",
        "barrier",
        "volatile",
    },
    "collection_ops": {
        "sort",
        "filter",
        "merge",
        "flatten",
        "group",
        "partition",
        "collect",
        "stream",
        "reduce",
        "aggregate",
    },
    "date_time": {
        "date",
        "time",
        "calendar",
        "temporal",
        "instant",
        "duration",
        "period",
        "zone",
        "epoch",
        "timestamp",
    },
    "serialization": {
        "serialize",
        "json",
        "xml",
        "marshal",
        "toJson",
        "fromJson",
        "toXml",
    },
    "comparison": {"compare", "diff", "deep", "shallow", "Comparator", "ordering"},
}


_ts_parser = None


def get_parser():
    global _ts_parser
    if _ts_parser is None:
        lang = tree_sitter.Language(tsjava.language())
        _ts_parser = tree_sitter.Parser(lang)
    return _ts_parser


def extract_invocations_and_types(body: str) -> tuple[list[str], list[str]]:
    parser = get_parser()
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
            if (
                method_name
                and method_name not in NOISE_INVOCATIONS
                and len(method_name) > 2
            ):
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


def split_camel_case(name: str) -> list[str]:
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", parts)
    return [p.lower() for p in parts.split() if len(p) > 1]


def extract_javadoc_summary(javadoc: str | None) -> str | None:
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
        "",
        first_sent,
    )
    first_sent = first_sent.strip()
    if len(first_sent) < 8:
        return None
    return first_sent[0].lower() + first_sent[1:]


def extract_param_types(signature: str) -> list[str]:
    m = re.search(r"\(([^)]*)\)", signature)
    if not m:
        return []
    params = m.group(1)
    types = re.findall(r"\b([A-Z][a-zA-Z0-9]+)\b", params)
    return [t for t in types if t not in NOISE_TYPES and len(t) > 2]


def extract_return_type(signature: str) -> str | None:
    m = re.match(r".*?\b([A-Z][a-zA-Z0-9<>,\s]+?)\s+\w+\s*\(", signature)
    if m:
        ret = m.group(1).strip()
        ret = re.sub(r"<.*>", "", ret).strip()
        if ret not in NOISE_TYPES and ret not in {
            "public",
            "private",
            "protected",
            "static",
            "final",
            "abstract",
        }:
            return ret
    return None


def infer_semantic_tags(
    method_tokens: list[str], invocations: list[str], types: list[str]
) -> list[str]:
    all_terms = set()
    for t in method_tokens:
        all_terms.add(t.lower())
    for inv in invocations:
        for part in split_camel_case(inv):
            all_terms.add(part.lower())
    for typ in types:
        all_terms.add(typ)
        for part in split_camel_case(typ):
            all_terms.add(part.lower())

    tags = []
    for tag, keywords in TAG_RULES.items():
        if all_terms & {k.lower() for k in keywords}:
            tags.append(tag)
    return tags[:3]


def classify_type(type_name: str) -> str:
    if type_name in JAVA_API_TYPES:
        return "api"
    return "project"


@dataclass
class MethodProfile:
    chunk: object
    class_name: str
    method_name: str
    file_path: str
    package_name: str
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

    method_tokens = split_camel_case(chunk.method_name)
    if len(method_tokens) < 1:
        return None

    class_tokens = split_camel_case(chunk.class_name) if chunk.class_name else []

    invocations, types = extract_invocations_and_types(chunk.body)

    if len(invocations) < 1 and not chunk.javadoc and len(method_tokens) < 2:
        return None

    javadoc_summary = extract_javadoc_summary(chunk.javadoc)
    param_types = extract_param_types(chunk.signature) if chunk.signature else []
    return_type = extract_return_type(chunk.signature) if chunk.signature else None
    is_test = "src/test" in chunk.file_path
    package_parts = chunk.file_path.replace("src/main/java/", "").replace(
        "src/test/java/", ""
    )
    package_name = "/".join(package_parts.split("/")[:-1])

    semantic_tags = infer_semantic_tags(method_tokens, invocations, types)

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
        package_name=package_name,
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


def query_behavioral(profile: MethodProfile) -> str | None:
    if len(profile.method_tokens) < 2:
        return None
    action = profile.method_tokens[0]
    obj_parts = profile.method_tokens[1:]
    obj = " ".join(obj_parts[:3])

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


def query_navigation(profile: MethodProfile) -> str | None:
    if profile.javadoc_summary:
        summary = profile.javadoc_summary
        words = summary.split()[:8]
        if len(words) >= 3:
            return " ".join(words)

    if profile.class_name and profile.semantic_tags:
        tag = profile.semantic_tags[0].replace("_", " ")
        return f"{tag} in {profile.class_name}"

    if profile.class_name and len(profile.method_tokens) >= 2:
        concept = " ".join(profile.method_tokens[:3])
        return f"find {concept} in {profile.class_name}"

    return None


def query_short(profile: MethodProfile) -> str | None:
    candidates: list[str] = []
    for t in profile.method_tokens:
        if t.lower() not in STOPWORDS and len(t) > 2:
            candidates.append(t)

    for inv in profile.top_invocations[:2]:
        for part in split_camel_case(inv):
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
    selected = unique[:count]
    return " ".join(selected)


def query_type_aware(profile: MethodProfile) -> str | None:
    parts = []
    for inv in profile.top_invocations[:2]:
        parts.extend(split_camel_case(inv))
    for t in profile.top_types[:2]:
        if t not in NOISE_TYPES:
            parts.extend(split_camel_case(t))

    seen = set()
    unique: list[str] = []
    for p in parts:
        pl = p.lower()
        if pl not in seen and pl not in STOPWORDS and len(pl) > 2:
            seen.add(pl)
            unique.append(p)

    if len(unique) < 2:
        return None
    return " ".join(unique[:4])


def build_target(profile: MethodProfile, query: str) -> dict:
    raw_keywords = tokenize(query)
    keywords = [k for k in raw_keywords if k.lower() not in STOPWORDS]
    keywords = list(dict.fromkeys(keywords))[:6]

    project_terms: list[str] = []
    api_hints: list[str] = []

    if profile.class_name:
        if classify_type(profile.class_name) == "api":
            api_hints.append(profile.class_name)
        else:
            project_terms.append(profile.class_name)

    for t in profile.top_types:
        if classify_type(t) == "api":
            if t not in api_hints:
                api_hints.append(t)
        else:
            if t not in project_terms:
                project_terms.append(t)

    for pt in profile.param_types:
        if classify_type(pt) == "api":
            if pt not in api_hints:
                api_hints.append(pt)
        else:
            if pt not in project_terms:
                project_terms.append(pt)

    project_terms = project_terms[:4]
    api_hints = api_hints[:4]

    method_hints = [profile.method_name]
    for inv in profile.top_invocations[:2]:
        if inv != profile.method_name:
            method_hints.append(inv)
    method_hints = method_hints[:3]

    search_queries: list[str] = []
    if keywords:
        search_queries.append(" ".join(keywords[:5]))
    if profile.class_name and profile.method_name:
        search_queries.append(f"{profile.class_name} {profile.method_name}")

    type_parts: list[str] = []
    if profile.method_name:
        type_parts.append(profile.method_name)
    type_parts.extend(api_hints[:2])
    if type_parts and len(type_parts) >= 2:
        search_queries.append(" ".join(type_parts))

    if profile.semantic_tags and profile.method_tokens:
        tag = profile.semantic_tags[0].replace("_", " ")
        action = " ".join(profile.method_tokens[:2])
        beh = f"{tag} {action}"
        search_queries.append(beh)
    search_queries = deduplicate_queries(search_queries)

    return {
        "intent": "find_code",
        "search_scope": "tests" if profile.is_test else "implementation",
        "keywords": keywords,
        "project_terms": project_terms,
        "method_hints": method_hints,
        "api_hints": api_hints,
        "search_queries": search_queries,
    }


def deduplicate_queries(queries: list[str]) -> list[str]:
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


def is_quality_sample(query: str, target: dict) -> bool:
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


def generate_samples(profiles: list[MethodProfile]) -> tuple[list[dict], dict]:
    generators = [
        ("behavioral", query_behavioral),
        ("navigation", query_navigation),
        ("short", query_short),
        ("type_aware", query_type_aware),
    ]

    samples = []
    style_counts = {name: 0 for name, _ in generators}
    filtered_count = 0

    for profile in profiles:
        for style_name, gen_fn in generators:
            query = gen_fn(profile)
            if query is None:
                continue

            target = build_target(profile, query)

            if not is_quality_sample(query, target):
                filtered_count += 1
                continue

            prompt = SYSTEM_PROMPT.format(query=query)
            completion = json.dumps(target, ensure_ascii=False)

            samples.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "style": style_name,
                    "method": profile.method_name,
                    "class": profile.class_name,
                    "file": profile.file_path,
                    "is_test": profile.is_test,
                    "semantic_tags": profile.semantic_tags,
                    "javadoc_summary": profile.javadoc_summary,
                    "body": profile.chunk.body,
                    "javadoc": profile.chunk.javadoc,
                }
            )
            style_counts[style_name] += 1

    stats = {
        "style_counts": style_counts,
        "filtered": filtered_count,
        "total": len(samples),
    }
    return samples, stats


def main():
    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks for {REPO_ID}...")
    chunks = load_chunks(REPO_ID)
    print(f"Loaded {len(chunks)} chunks")

    print("\nBuilding method profiles with tree-sitter...")
    profiles = []
    for chunk in chunks:
        profile = build_profile(chunk)
        if profile is not None:
            profiles.append(profile)

    print(f"Built {len(profiles)} profiles")
    test_profiles = [p for p in profiles if p.is_test]
    impl_profiles = [p for p in profiles if not p.is_test]
    print(f"Implementation: {len(impl_profiles)}, Test: {len(test_profiles)}")
    with_javadoc = [p for p in profiles if p.javadoc_summary]
    print(f"With javadoc: {len(with_javadoc)}")
    with_tags = [p for p in profiles if p.semantic_tags]
    print(f"With semantic tags: {len(with_tags)}")

    print("\nSelecting quality methods...")
    profiles.sort(key=lambda p: p.quality_score, reverse=True)

    max_test = int(500 * 0.25)
    max_impl = 500 - max_test
    selected_impl = [p for p in profiles if not p.is_test][:max_impl]
    selected_test = [p for p in profiles if p.is_test][:max_test]
    selected = selected_impl + selected_test
    random.shuffle(selected)
    print(
        f"Selected {len(selected)} methods ({len(selected_impl)} impl, {len(selected_test)} test)"
    )

    print("\nGenerating training samples...")
    samples, stats = generate_samples(selected)
    print(f"Generated {stats['total']} samples (filtered {stats['filtered']})")
    for style, count in stats["style_counts"].items():
        print(f"{style}: {count}")

    # Dataset-level stats
    test_samples = [s for s in samples if s["is_test"]]
    test_ratio = len(test_samples) / len(samples) * 100 if samples else 0
    print(
        f"\n  Test sample ratio: {len(test_samples)}/{len(samples)} ({test_ratio:.1f}%)"
    )

    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train = samples[:split]
    val = samples[split:]

    train_path = DATA_DIR / "train_rewriter.jsonl"
    val_path = DATA_DIR / "val_rewriter.jsonl"

    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w") as f:
            for sample in data:
                out = {
                    "prompt": sample["prompt"],
                    "completion": sample["completion"],
                    "style": sample["style"],
                    "method": sample["method"],
                    "class": sample["class"],
                    "file": sample["file"],
                    "is_test": sample["is_test"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(data)} samples to {path}")

    print("\nGenerating DATASET_EXAMPLES.md...")
    generate_examples_report(samples)

    print("\n" + "=" * 60)
    print("SAMPLE EXAMPLES")
    print("=" * 60)
    for i, s in enumerate(samples[:8]):
        target = json.loads(s["completion"])
        print(f"\n--- Example {i + 1} (style: {s['style']}) ---")
        print(f"Method: {s['class']}.{s['method']}")
        print(f"Tags: {s['semantic_tags']}")
        q_start = s["prompt"].index("Query: ") + 7
        q_end = s["prompt"].index("\n\nOutput")
        print(f"Query: {s['prompt'][q_start:q_end]}")
        print(f"Target: {json.dumps(target, indent=2)[:400]}")

    print(f"\nTrain: {len(train)}, Val: {len(val)}")


def generate_examples_report(samples: list[dict]):
    by_style = {}
    for s in samples:
        by_style.setdefault(s["style"], []).append(s)

    lines = []
    lines.append("# Dataset Examples Report (v2 - Improved Generator)")
    lines.append("")
    lines.append(f"**Total samples:** {len(samples)}")
    lines.append(f"**Repository:** jdereg/java-util")
    lines.append(f"**Styles:** {list(by_style.keys())}")
    lines.append("")
    lines.append("## Style Distribution")
    lines.append("")
    for style, items in sorted(by_style.items()):
        test_count = sum(1 for s in items if s["is_test"])
        lines.append(f"- **{style}:** {len(items)} samples ({test_count} test)")
    lines.append("")

    for style, items in sorted(by_style.items()):
        lines.append(f"## Style: {style}")
        lines.append("")
        picked = random.sample(items, min(5, len(items)))
        for i, s in enumerate(picked):
            target = json.loads(s["completion"])
            lines.append(f"### Example {i + 1}")
            lines.append("")
            lines.append(f"**Class:** `{s['class']}`")
            lines.append(f"**Method:** `{s['method']}`")
            lines.append(f"**File:** `{s['file']}`")
            lines.append(f"**Is test:** {s['is_test']}")
            lines.append(f"**Semantic tags:** {s.get('semantic_tags', [])}")
            lines.append("")

            if s.get("javadoc"):
                lines.append("**Javadoc:**")
                lines.append("```java")
                lines.append(s["javadoc"][:500])
                lines.append("```")
                lines.append("")

            if s.get("body"):
                lines.append("**Method body:**")
                lines.append("```java")
                lines.append(s["body"][:1000])
                if len(s["body"]) > 1000:
                    lines.append("// ... (truncated)")
                lines.append("```")
                lines.append("")

            if s.get("javadoc_summary"):
                lines.append(f"**Javadoc summary:** {s['javadoc_summary']}")
                lines.append("")

            lines.append("**Input (prompt):**")
            lines.append("```")
            lines.append(s["prompt"])
            lines.append("```")
            lines.append("")
            lines.append("**Target (completion):**")
            lines.append("```json")
            lines.append(json.dumps(target, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    report_path = Path(__file__).parent / "DATASET_EXAMPLES.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved to {report_path}")


if __name__ == "__main__":
    main()
