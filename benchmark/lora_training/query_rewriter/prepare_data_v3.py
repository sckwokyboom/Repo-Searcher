import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "backend"))

from llm_client import batch_generate_json_arrays, unload_model
from prepare_data import (
    SYSTEM_PROMPT,
    MethodProfile,
    build_profile,
    build_target,
    deduplicate_queries,
    is_quality_sample,
    query_behavioral,
    query_navigation,
    query_short,
    query_type_aware,
)

from backend.app.indexer.store import load_chunks

REPO_ID = "jdereg--java-util"
DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent / "data" / "cache"

MAX_PER_CLASS = 20
MAX_PER_METHOD = 3
MAX_PER_SEMANTIC_BUCKET = 5
TEST_RATIO_CAP = 0.15
VAL_PROFILE_RATIO = 0.12

QUERY_GEN_SYSTEM = (
    "You generate realistic code search queries that a developer would type "
    "when searching for a specific piece of Java code in a codebase. "
    "Output 2-3 short queries (3-8 words each), each from a different angle: "
    "one describing behavior, one for navigation, one with keywords. "
    "Output ONLY a JSON array of strings, nothing else."
)


def build_llm_prompt(profile: MethodProfile) -> str:
    param_summary = ""
    if profile.param_types:
        param_summary = ", ".join(profile.param_types[:4])

    ret = profile.return_type or "void"

    javadoc = profile.javadoc_summary or "N/A"

    tags = ", ".join(profile.semantic_tags) if profile.semantic_tags else "general"

    calls = ", ".join(profile.top_invocations[:3]) if profile.top_invocations else "N/A"

    types = ", ".join(profile.top_types[:3]) if profile.top_types else "N/A"

    return (
        f"Class: {profile.class_name}\n"
        f"Method: {profile.method_name}({param_summary})\n"
        f"Returns: {ret}\n"
        f"Purpose: {javadoc}\n"
        f"Tags: {tags}\n"
        f"Calls: {calls}\n"
        f"Types: {types}"
    )


def generate_llm_queries(profiles: list[MethodProfile]) -> dict[int, list[str]]:
    """Generate LLM-naturalized queries for all profiles. Returns {profile_index: [queries]}."""
    prompts = [build_llm_prompt(p) for p in profiles]

    print(f"\nGenerating LLM queries for {len(profiles)} profiles...")
    results = batch_generate_json_arrays(
        prompts,
        system=QUERY_GEN_SYSTEM,
        temperature=0.7,
        max_new_tokens=200,
        cache_path=CACHE_DIR / "llm_queries_cache.json",
        progress_every=20,
    )

    queries_map: dict[int, list[str]] = {}
    success = 0
    fallback_count = 0

    for i, parsed in enumerate(results):
        if parsed and len(parsed) >= 2:
            # Filter queries: 3-8 words, not too long
            filtered = []
            for q in parsed:
                q = q.strip()
                words = q.split()
                if 3 <= len(words) <= 10 and len(q) <= 100:
                    filtered.append(q.lower() if q[0].isupper() and " " in q else q)
            if len(filtered) >= 2:
                queries_map[i] = filtered[:3]
                success += 1
                continue

        profile = profiles[i]
        fallback_queries = []
        for gen_fn in [
            query_behavioral,
            query_navigation,
            query_short,
            query_type_aware,
        ]:
            q = gen_fn(profile)
            if q:
                fallback_queries.append(q)
            if len(fallback_queries) >= 2:
                break
        if fallback_queries:
            queries_map[i] = fallback_queries[:3]
            fallback_count += 1

    print(f"LLM success: {success}/{len(profiles)}")
    print(f"Fallback: {fallback_count}/{len(profiles)}")
    print(f"No queries: {len(profiles) - success - fallback_count}/{len(profiles)}")

    return queries_map


def class_balanced_sample(samples: list[dict]) -> list[dict]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_class[s["class"]].append(s)

    balanced = []

    for cls, cls_samples in by_class.items():
        by_method: dict[str, list[dict]] = defaultdict(list)
        for s in cls_samples:
            by_method[s["method"]].append(s)

        method_capped = []
        for method, method_samples in by_method.items():
            random.shuffle(method_samples)
            method_capped.extend(method_samples[:MAX_PER_METHOD])

        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for s in method_capped:
            bucket_key = str(sorted(s.get("semantic_tags", [])))
            by_bucket[bucket_key].append(s)

        bucket_capped = []
        for bucket, bucket_samples in by_bucket.items():
            bucket_capped.extend(bucket_samples[:MAX_PER_SEMANTIC_BUCKET])

        random.shuffle(bucket_capped)
        balanced.extend(bucket_capped[:MAX_PER_CLASS])

    return balanced


def enforce_test_ratio(
    samples: list[dict], max_ratio: float = TEST_RATIO_CAP
) -> list[dict]:
    """Ensure test samples don't exceed max_ratio of total."""
    test_samples = [s for s in samples if s.get("is_test")]
    impl_samples = [s for s in samples if not s.get("is_test")]

    max_test = int(len(impl_samples) * max_ratio / (1 - max_ratio))
    if len(test_samples) > max_test:
        random.shuffle(test_samples)
        test_samples = test_samples[:max_test]

    return impl_samples + test_samples


def profile_level_split(
    profiles: list[MethodProfile],
    queries_map: dict[int, list[str]],
    val_ratio: float = VAL_PROFILE_RATIO,
) -> tuple[list[int], list[int]]:
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(profiles):
        if i in queries_map:
            by_class[p.class_name].append(i)

    train_indices = []
    val_indices = []

    for cls, indices in by_class.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        if len(indices) <= 2:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

    return train_indices, val_indices


def generate_samples(
    profiles: list[MethodProfile],
    queries_map: dict[int, list[str]],
    profile_indices: list[int],
) -> list[dict]:
    samples = []

    for idx in profile_indices:
        if idx not in queries_map:
            continue
        profile = profiles[idx]
        queries = queries_map[idx]

        for query in queries:
            target = build_target(profile, query)

            if not is_quality_sample(query, target):
                continue

            prompt = SYSTEM_PROMPT.format(query=query)
            completion = json.dumps(target, ensure_ascii=False)

            samples.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "method": profile.method_name,
                    "class": profile.class_name,
                    "file": profile.file_path,
                    "is_test": profile.is_test,
                    "semantic_tags": profile.semantic_tags,
                    "query_source": "llm",
                }
            )

    return samples


def main():
    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(REPO_ID)
    print(f"Loaded {len(chunks)} chunks")

    profiles = []
    for chunk in chunks:
        profile = build_profile(chunk)
        if profile is not None:
            profiles.append(profile)

    impl_profiles = [p for p in profiles if not p.is_test]
    test_profiles = [p for p in profiles if p.is_test]
    print(f"Implementation: {len(impl_profiles)}, Test: {len(test_profiles)}")

    profiles.sort(key=lambda p: p.quality_score, reverse=True)

    target_profiles = 450
    max_test = int(target_profiles * TEST_RATIO_CAP)
    max_impl = target_profiles - max_test

    selected_impl = [p for p in profiles if not p.is_test][:max_impl]
    selected_test = [p for p in profiles if p.is_test][:max_test]
    selected = selected_impl + selected_test
    random.shuffle(selected)

    class_counts = defaultdict(int)
    for p in selected:
        class_counts[p.class_name] += 1
    top_classes = sorted(class_counts.items(), key=lambda x: -x[1])[:15]
    for cls, cnt in top_classes:
        print(f"{cls}: {cnt}")

    queries_map = generate_llm_queries(selected)
    unload_model()

    train_indices, val_indices = profile_level_split(selected, queries_map)

    train_samples = generate_samples(selected, queries_map, train_indices)
    print(f"Raw train samples: {len(train_samples)}")

    val_samples = generate_samples(selected, queries_map, val_indices)
    print(f"Raw val samples: {len(val_samples)}")

    train_samples = class_balanced_sample(train_samples)
    print(f"After class balancing: {len(train_samples)}")

    train_samples = enforce_test_ratio(train_samples)
    print(f"After test ratio enforcement: {len(train_samples)}")

    for label, samples in [("Train", train_samples), ("Val", val_samples)]:
        test_count = sum(1 for s in samples if s["is_test"])
        test_pct = test_count / len(samples) * 100 if samples else 0

        class_dist = defaultdict(int)
        for s in samples:
            class_dist[s["class"]] += 1

        method_dist = defaultdict(int)
        for s in samples:
            method_dist[s["method"]] += 1

        print(f"\n{label}: {len(samples)} samples")
        print(f"Test ratio: {test_count}/{len(samples)} ({test_pct:.1f}%)")
        print(f"Unique classes: {len(class_dist)}")
        print(f"Unique methods: {len(method_dist)}")
        print(f"Max per class: {max(class_dist.values()) if class_dist else 0}")
        print(f"Max per method: {max(method_dist.values()) if method_dist else 0}")

        top5 = sorted(class_dist.items(), key=lambda x: -x[1])[:5]
        print(f"Top 5 classes: {top5}")

    random.shuffle(train_samples)

    train_path = DATA_DIR / "train_rewriter_v3.jsonl"
    val_path = DATA_DIR / "val_rewriter_v3.jsonl"

    for path, samples in [(train_path, train_samples), (val_path, val_samples)]:
        with open(path, "w") as f:
            for s in samples:
                out = {
                    "prompt": s["prompt"],
                    "completion": s["completion"],
                    "method": s["method"],
                    "class": s["class"],
                    "file": s["file"],
                    "is_test": s["is_test"],
                    "query_source": s.get("query_source", "unknown"),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(samples)} samples to {path}")

    for i, s in enumerate(train_samples[:10]):
        target = json.loads(s["completion"])
        q_start = s["prompt"].index("Query: ") + 7
        q_end = s["prompt"].index("\n\nOutput")
        query = s["prompt"][q_start:q_end]
        print(f"\n--- Example {i + 1} ---")
        print(f"Method: {s['class']}.{s['method']}")
        print(f"Query:  {query}")
        print(f"Terms:  {target.get('project_terms', [])}")
        print(f"Hints:  {target.get('method_hints', [])}")
        print(f"Search: {target.get('search_queries', [])[:2]}")


if __name__ == "__main__":
    main()
