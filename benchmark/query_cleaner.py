"""Clean commit messages to extract meaningful search queries."""

import re


def clean_query(description: str) -> tuple[str, bool]:
    """
    Clean a commit message into a search query.
    Keeps all useful content, removes only boilerplate (Co-authored-by, Generated with, etc.).
    Returns (cleaned_query, is_low_quality).
    """
    if not description:
        return "", True

    lines = description.strip().split('\n')
    # Keep all non-boilerplate lines
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if _is_boilerplate(stripped):
            continue
        if stripped:
            cleaned_lines.append(stripped)

    query = ' '.join(cleaned_lines)

    # Remove conventional commit prefixes (only when followed by colon)
    query = re.sub(r'^(feat|fix|refactor|docs|style|test|chore|perf|ci|build|revert)(\(.+?\))?\s*:\s*', '', query, flags=re.IGNORECASE)

    # Remove SHA references
    query = re.sub(r'\b[0-9a-f]{7,40}\b', '', query)

    # Remove markdown link syntax but keep text
    query = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', query)

    # Remove PR/issue references like (#1413)
    query = re.sub(r'\s*\(#\d+\)', '', query)

    # Remove emoji (common in commit messages)
    query = re.sub(r'[\U0001F300-\U0001F9FF]', '', query)

    # Remove extra whitespace
    query = re.sub(r'\s+', ' ', query).strip()

    # Check quality
    word_count = len(query.split())
    is_low_quality = word_count < 5

    return query, is_low_quality


def _is_boilerplate(line: str) -> bool:
    """Check if a line is boilerplate that should be removed."""
    lower = line.lower()
    boilerplate_patterns = [
        'co-authored-by:',
        'co-authored by:',
        'generated with claude',
        'generated with [claude',
        'signed-off-by:',
        'reviewed-by:',
        'acked-by:',
    ]
    return any(p in lower for p in boilerplate_patterns)
