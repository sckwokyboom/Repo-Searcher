from pathlib import Path

import networkx as nx
import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser

from app.models.search import CodeChunk

JAVA_LANGUAGE = Language(tsjava.language())


def _get_node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode(
        "utf-8", errors="replace"
    )


def _find_containing_method(node: Node) -> str | None:
    current = node.parent
    while current:
        if current.type in ("method_declaration", "constructor_declaration"):
            class_name = None
            method_name = None
            parent = current.parent
            while parent:
                if parent.type in ("class_declaration", "interface_declaration"):
                    for child in parent.children:
                        if child.type == "identifier" and child.text is not None:
                            class_name = child.text.decode("utf-8")
                            break
                    break
                parent = parent.parent

            for child in current.children:
                if child.type == "identifier" and child.text is not None:
                    method_name = child.text.decode("utf-8")
                    break

            if current.type == "constructor_declaration":
                method_name = class_name or "constructor"

            if method_name and class_name:
                return f"{class_name}.{method_name}"
            return method_name
        current = current.parent
    return None


def build_call_graph(chunks: list[CodeChunk], repo_path: Path) -> nx.DiGraph:
    graph = nx.DiGraph()

    method_name_to_chunks: dict[str, list[str]] = {}
    for chunk in chunks:
        if chunk.chunk_type == "method" and chunk.method_name:
            graph.add_node(
                chunk.chunk_id, label=chunk.method_name, file_path=chunk.file_path
            )
            method_name_to_chunks.setdefault(chunk.method_name, []).append(
                chunk.chunk_id
            )

    parser = Parser(JAVA_LANGUAGE)
    java_files = sorted(repo_path.rglob("*.java"))

    for file_path in java_files:
        try:
            source = file_path.read_bytes()
            tree = parser.parse(source)
            relative_path = str(file_path.relative_to(repo_path))

            _extract_calls(
                tree.root_node,
                source,
                relative_path,
                graph,
                method_name_to_chunks,
                chunks,
            )
        except Exception:
            continue

    return graph


def _extract_calls(
    node: Node,
    source: bytes,
    file_path: str,
    graph: nx.DiGraph,
    method_name_to_chunks: dict[str, list[str]],
    chunks: list[CodeChunk],
):
    if node.type == "method_invocation":
        for child in node.children:
            if child.type == "identifier" and child.text is not None:
                invoked_name = child.text.decode("utf-8")
                caller = _find_containing_method(node)

                if caller and invoked_name in method_name_to_chunks:
                    caller_chunk_ids = [
                        c.chunk_id
                        for c in chunks
                        if c.file_path == file_path
                        and c.method_name == caller.split(".")[-1]
                    ]
                    if not caller_chunk_ids:
                        caller_chunk_ids = [
                            c.chunk_id
                            for c in chunks
                            if c.method_name == caller.split(".")[-1]
                        ]

                    target_ids = method_name_to_chunks[invoked_name]
                    same_file = [t for t in target_ids if t.startswith(file_path)]
                    target = same_file[0] if same_file else target_ids[0]

                    for caller_id in caller_chunk_ids[:1]:
                        if caller_id != target:
                            graph.add_edge(caller_id, target)
                break

    for child in node.children:
        _extract_calls(child, source, file_path, graph, method_name_to_chunks, chunks)
