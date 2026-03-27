import re

from rank_bm25 import BM25Okapi

from app.models.search import CodeChunk


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.split(r"[\s\.\,\;\:\(\)\{\}\[\]\"\']+", text):
        if not word:
            continue
        camel_parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", word)
        camel_parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", camel_parts)
        for part in camel_parts.split():
            if len(part) > 1:
                tokens.append(part.lower())
        for part in word.split("_"):
            if len(part) > 1:
                tokens.append(part.lower())
    return tokens


def build_bm25_index(
    chunks: list[CodeChunk],
) -> tuple[BM25Okapi, list[list[str]]]:
    tokenized_corpus = [tokenize(chunk.text_representation) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus
