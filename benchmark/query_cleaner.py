import re


def clean_query(description: str) -> tuple[str, bool]:
    if not description:
        return "", True

    lines = description.strip().split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if _is_boilerplate(stripped):
            continue
        if stripped:
            cleaned_lines.append(stripped)

    query = " ".join(cleaned_lines)

    query = re.sub(
        r"^(feat|fix|refactor|docs|style|test|chore|perf|ci|build|revert)(\(.+?\))?\s*:\s*",
        "",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(r"\b[0-9a-f]{7,40}\b", "", query)
    query = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", query)
    query = re.sub(r"\s*\(#\d+\)", "", query)
    query = re.sub(r"[\U0001F300-\U0001F9FF]", "", query)
    query = re.sub(r"\s+", " ", query).strip()
    word_count = len(query.split())
    is_low_quality = word_count < 5

    return query, is_low_quality


def _is_boilerplate(line: str) -> bool:
    lower = line.lower()
    boilerplate_patterns = [
        "co-authored-by:",
        "co-authored by:",
        "generated with claude",
        "generated with [claude",
        "signed-off-by:",
        "reviewed-by:",
        "acked-by:",
    ]
    return any(p in lower for p in boilerplate_patterns)
