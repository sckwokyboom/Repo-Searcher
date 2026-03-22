import re
from pathlib import Path
from typing import Awaitable, Callable

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from app.models.search import CodeChunk

JAVA_LANGUAGE = Language(tsjava.language())


def _create_parser() -> Parser:
    parser = Parser(JAVA_LANGUAGE)
    return parser


def _get_node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_javadoc(node, source_bytes: bytes) -> str | None:
    prev = node.prev_named_sibling
    if prev and prev.type == "block_comment":
        text = _get_node_text(prev, source_bytes)
        if text.startswith("/**"):
            return text
    return None


def _extract_method_signature(node, source_bytes: bytes) -> str:
    parts = []
    for child in node.children:
        if child.type == "block" or child.type == "constructor_body":
            break
        parts.append(_get_node_text(child, source_bytes))
    return " ".join(parts)


def _extract_class_name_from_ancestors(node) -> str | None:
    current = node.parent
    while current:
        if current.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            for child in current.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")
        current = current.parent
    return None


def parse_file(file_path: Path, repo_root: Path) -> list[CodeChunk]:
    source = file_path.read_bytes()
    parser = _create_parser()
    tree = parser.parse(source)

    relative_path = str(file_path.relative_to(repo_root))
    chunks = []

    def visit(node):
        if node.type in ("method_declaration", "constructor_declaration"):
            name_node = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break

            if node.type == "constructor_declaration":
                method_name = _extract_class_name_from_ancestors(node) or "constructor"
            elif name_node:
                method_name = _get_node_text(name_node, source)
            else:
                method_name = "unknown"

            class_name = _extract_class_name_from_ancestors(node)
            javadoc = _find_javadoc(node, source)
            signature = _extract_method_signature(node, source)
            body = _get_node_text(node, source)

            text_parts = [method_name]
            if javadoc:
                text_parts.append(javadoc)
            text_parts.append(signature)
            text_parts.append(body)

            chunk_id = f"{relative_path}::{class_name}.{method_name}" if class_name else f"{relative_path}::{method_name}"

            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    chunk_type="method",
                    file_path=relative_path,
                    class_name=class_name,
                    method_name=method_name,
                    signature=signature,
                    javadoc=javadoc,
                    body=body,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    text_representation="\n".join(text_parts),
                )
            )

        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return chunks


async def parse_repository(
    repo_path: Path,
    progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
) -> tuple[list[CodeChunk], int]:
    java_files = sorted(repo_path.rglob("*.java"))
    all_chunks = []

    for i, file_path in enumerate(java_files):
        try:
            file_chunks = parse_file(file_path, repo_path)
            all_chunks.extend(file_chunks)
        except Exception:
            pass

        if progress_callback and (i % 5 == 0 or i == len(java_files) - 1):
            await progress_callback(i + 1, len(java_files))

    return all_chunks, len(java_files)
