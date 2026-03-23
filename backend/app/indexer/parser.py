import re
from pathlib import Path
from typing import Awaitable, Callable

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from app.models.search import CodeChunk

JAVA_LANGUAGE = Language(tsjava.language())

# Minimum meaningful body length (skip trivially small chunks)
MIN_BODY_LINES = 2
MIN_BODY_CHARS = 30


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


def _has_method_body(node) -> bool:
    """Check if a method_declaration actually has a { block } body.
    Interface methods and abstract methods don't — they end with ';'."""
    for child in node.children:
        if child.type in ("block", "constructor_body"):
            return True
    return False


def _extract_class_name_from_ancestors(node) -> str | None:
    current = node.parent
    while current:
        if current.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            for child in current.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")
        current = current.parent
    return None


def _extract_class_javadoc_from_ancestors(node, source_bytes: bytes) -> str | None:
    """Find the javadoc of the containing class for context enrichment."""
    current = node.parent
    while current:
        if current.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            doc = _find_javadoc(current, source_bytes)
            return doc
        current = current.parent
    return None


def _extract_class_signature(node, source_bytes: bytes) -> str:
    """Extract class declaration signature (modifiers, name, extends, implements)."""
    parts = []
    for child in node.children:
        if child.type in ("class_body", "interface_body", "enum_body"):
            break
        parts.append(_get_node_text(child, source_bytes))
    return " ".join(parts)


def _extract_class_members_summary(node, source_bytes: bytes) -> str:
    """Extract a summary of class fields and method signatures (no bodies)."""
    fields = []
    method_sigs = []

    body_node = None
    for child in node.children:
        if child.type in ("class_body", "interface_body", "enum_body"):
            body_node = child
            break

    if not body_node:
        return ""

    for member in body_node.children:
        if member.type == "field_declaration":
            field_text = _get_node_text(member, source_bytes).strip()
            if field_text:
                fields.append(field_text)
        elif member.type in ("method_declaration", "constructor_declaration"):
            sig = _extract_method_signature(member, source_bytes).strip()
            if sig:
                method_sigs.append(sig)

    parts = []
    if fields:
        parts.append("Fields:\n  " + "\n  ".join(fields[:20]))
    if method_sigs:
        parts.append("Methods:\n  " + "\n  ".join(method_sigs[:30]))
    return "\n".join(parts)


def _is_in_interface(node) -> bool:
    """Check if a node is inside an interface declaration."""
    current = node.parent
    while current:
        if current.type == "interface_declaration":
            return True
        if current.type == "class_declaration":
            return False
        current = current.parent
    return False


def _is_valid_method_chunk(body: str) -> bool:
    """Validate that a method body is a meaningful code chunk, not a fragment."""
    stripped = body.strip()
    if not stripped:
        return False
    if len(stripped) < MIN_BODY_CHARS:
        return False
    lines = [l for l in stripped.split("\n") if l.strip()]
    if len(lines) < MIN_BODY_LINES:
        return False
    return True


def _clean_body(body: str) -> str:
    """Clean up a method body: normalize whitespace, remove trailing blank lines."""
    lines = body.split("\n")
    # Remove leading/trailing empty lines but keep internal structure
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def parse_file(file_path: Path, repo_root: Path) -> list[CodeChunk]:
    source = file_path.read_bytes()
    parser = _create_parser()
    tree = parser.parse(source)

    relative_path = str(file_path.relative_to(repo_root))
    chunks = []

    # First pass: collect class-level info for method enrichment
    class_info: dict[str, dict] = {}  # class_name -> {javadoc, name}

    def collect_classes(node):
        if node.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            name = None
            for child in node.children:
                if child.type == "identifier":
                    name = child.text.decode("utf-8")
                    break
            if name:
                javadoc = _find_javadoc(node, source)
                class_info[name] = {
                    "javadoc": javadoc,
                    "name": name,
                }
        for child in node.children:
            collect_classes(child)

    collect_classes(tree.root_node)

    def visit(node):
        # --- Class-level chunks ---
        if node.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            class_name = None
            for child in node.children:
                if child.type == "identifier":
                    class_name = child.text.decode("utf-8")
                    break

            if class_name:
                javadoc = _find_javadoc(node, source)
                signature = _extract_class_signature(node, source)
                members_summary = _extract_class_members_summary(node, source)

                # Build class body: signature + fields + method sigs
                # This gives a complete overview even if members_summary is sparse
                body_parts = [signature + " {"]
                if members_summary:
                    body_parts.append(members_summary)
                body_parts.append("}")
                class_body = "\n".join(body_parts)

                text_parts = [class_name]
                if javadoc:
                    text_parts.append(javadoc)
                text_parts.append(signature)
                if members_summary:
                    text_parts.append(members_summary)

                chunk_id = f"{relative_path}::{class_name}"

                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        chunk_type="class",
                        file_path=relative_path,
                        class_name=class_name,
                        method_name=None,
                        signature=signature,
                        javadoc=javadoc,
                        body=class_body,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        text_representation="\n".join(text_parts),
                    )
                )

        # --- Method-level chunks (enriched with class context) ---
        if node.type in ("method_declaration", "constructor_declaration"):
            # Skip interface method declarations (no body, just signature + ;)
            if not _has_method_body(node):
                # Still recurse into children
                for child in node.children:
                    visit(child)
                return

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

            # Clean up the body
            body = _clean_body(body)

            # Skip trivially small chunks (empty methods, single-return getters etc.)
            if not _is_valid_method_chunk(body):
                for child in node.children:
                    visit(child)
                return

            # Enriched text_representation with class context
            text_parts = []
            if class_name:
                text_parts.append(class_name)
                # Add short class javadoc for context
                ci = class_info.get(class_name)
                if ci and ci.get("javadoc"):
                    short_doc = ci["javadoc"][:200]
                    text_parts.append(short_doc)
            text_parts.append(method_name)
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
