import re

from app.ml.model_manager import get_model_manager


def expand_query(query: str) -> tuple[str, list[str]]:
    manager = get_model_manager()

    prompt = (
        "You are a code search assistant. Given a natural language query about Java code, "
        "generate 5-10 relevant search keywords including likely class names, method names, "
        "and technical terms that would appear in the source code.\n\n"
        f"Query: {query}\n\n"
        "Keywords (comma-separated):"
    )

    response = manager.generate(prompt, max_new_tokens=100)

    # Parse keywords
    keywords = []
    for part in re.split(r'[,\n]+', response):
        word = part.strip().strip('- "\'')
        if word and len(word) > 1 and len(word) < 50:
            keywords.append(word)

    keywords = keywords[:10]
    expanded = f"{query} {' '.join(keywords)}"

    return expanded, keywords
