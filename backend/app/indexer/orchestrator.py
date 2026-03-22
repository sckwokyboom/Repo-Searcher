import json
import logging
from pathlib import Path
from typing import Awaitable, Callable

from app.config import settings
from app.models.repo import IndexingProgress, IndexingStep, RepoInfo

logger = logging.getLogger(__name__)


class IndexingOrchestrator:
    def __init__(
        self,
        repo: RepoInfo,
        progress_callback: Callable[[IndexingProgress], Awaitable[None]],
    ):
        self.repo = repo
        self.callback = progress_callback

    async def _emit(
        self,
        step: IndexingStep,
        progress: float,
        message: str,
        files_processed: int = 0,
        files_total: int = 0,
    ):
        await self.callback(
            IndexingProgress(
                repo_id=self.repo.repo_id,
                step=step,
                progress=progress,
                message=message,
                files_processed=files_processed,
                files_total=files_total,
            )
        )

    async def run(self):
        try:
            # Step 1: Clone
            await self._emit(IndexingStep.CLONING, 0.0, "Cloning repository...")
            from app.indexer.cloner import clone_repo

            repo_path = await clone_repo(self.repo.url, self.repo.repo_id)
            await self._emit(IndexingStep.CLONING, 1.0, "Clone complete")

            # Step 2: Parse
            await self._emit(IndexingStep.PARSING, 0.0, "Parsing Java files...")
            from app.indexer.parser import parse_repository

            chunks, java_files_count = await parse_repository(
                repo_path,
                lambda processed, total: self._emit(
                    IndexingStep.PARSING,
                    processed / max(total, 1),
                    f"Parsed {processed}/{total} files",
                    files_processed=processed,
                    files_total=total,
                ),
            )
            await self._emit(
                IndexingStep.PARSING,
                1.0,
                f"Parsing complete: {len(chunks)} chunks from {java_files_count} files",
            )

            if not chunks:
                await self._emit(IndexingStep.FAILED, 0.0, "No Java files found in repository")
                return

            # Step 3: BM25
            await self._emit(IndexingStep.BUILDING_BM25, 0.0, "Building BM25 index...")
            from app.indexer.bm25_builder import build_bm25_index

            bm25_index, tokenized_corpus = build_bm25_index(chunks)
            await self._emit(IndexingStep.BUILDING_BM25, 1.0, "BM25 index built")

            # Step 4: Vectors
            await self._emit(IndexingStep.BUILDING_VECTORS, 0.0, "Building vector index...")
            from app.indexer.vector_builder import build_vector_index

            faiss_index = await build_vector_index(
                chunks,
                lambda progress: self._emit(
                    IndexingStep.BUILDING_VECTORS,
                    progress,
                    f"Encoding chunks... {int(progress * 100)}%",
                ),
            )
            await self._emit(IndexingStep.BUILDING_VECTORS, 1.0, "Vector index built")

            # Step 5: Call Graph
            await self._emit(IndexingStep.BUILDING_CALLGRAPH, 0.0, "Building call graph...")
            from app.indexer.callgraph_builder import build_call_graph

            call_graph = build_call_graph(chunks, repo_path)
            await self._emit(
                IndexingStep.BUILDING_CALLGRAPH,
                1.0,
                f"Call graph built: {call_graph.number_of_nodes()} nodes, {call_graph.number_of_edges()} edges",
            )

            # Step 6: Save
            await self._emit(IndexingStep.SAVING, 0.0, "Saving indexes...")
            from app.indexer.store import save_indexes

            self.repo.chunk_count = len(chunks)
            save_indexes(
                self.repo,
                chunks,
                bm25_index,
                tokenized_corpus,
                faiss_index,
                call_graph,
            )
            await self._emit(IndexingStep.SAVING, 1.0, "Indexes saved")

            # Register in registry
            self._register_repo()

            await self._emit(IndexingStep.DONE, 1.0, "Indexing complete!")

        except Exception as e:
            logger.exception("Indexing failed")
            await self._emit(IndexingStep.FAILED, 0.0, f"Error: {str(e)}")

    def _register_repo(self):
        registry_path = settings.indexes_dir / "registry.json"
        repos = []
        if registry_path.exists():
            with open(registry_path) as f:
                repos = json.load(f)

        repos = [r for r in repos if r["repo_id"] != self.repo.repo_id]

        from datetime import datetime, timezone

        self.repo.indexed_at = datetime.now(timezone.utc)
        repos.insert(0, self.repo.model_dump(mode="json"))

        with open(registry_path, "w") as f:
            json.dump(repos, f, indent=2, default=str)
