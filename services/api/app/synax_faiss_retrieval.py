# services/api/app/synax_faiss_retrieval.py
import os
import json
import threading
import faiss
import numpy as np
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from concurrent.futures import as_completed

from services.api.app.synax_config import (
    FAISS_SHARD_DIR,
    NUM_FAISS_SHARDS,
    FAISS_SEARCH_EXECUTOR,
)


@dataclass(slots=True)
class FaissSearchResult:
    vector_id: str
    document_id: str
    source: str
    embedding_type: str
    text: str
    similarity: float
    chunk_id: int = -1
    start_char: int = 0
    end_char: int = 0
    sentence_count: int = 0
    average_similarity: float = 0.0
    token_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "document_id": self.document_id,
            "source": self.source,
            "embedding_type": self.embedding_type,
            "text": self.text,
            "similarity": self.similarity,
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "sentence_count": self.sentence_count,
            "average_similarity": self.average_similarity,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FaissShard:
    shard_id: int
    index: faiss.Index
    metadata: Dict[str, Dict[str, Any]]


class FaissIndexLoader:
    def __init__(self, source: str):
        self.source = source
        self.source_dir = os.path.join(FAISS_SHARD_DIR, source)
        self.shards: List[FaissShard] = []
        self._loaded = False
        self._lock = threading.Lock()
        self._file_state: Dict[int, tuple] = {}

    def _get_file_state(self, shard_id: int) -> tuple:
        index_path = os.path.join(self.source_dir, f"shard_{shard_id}.index")
        metadata_path = os.path.join(self.source_dir, f"shard_{shard_id}.metadata.json")
        index_mtime = (
            os.stat(index_path).st_mtime_ns
            if os.path.exists(index_path)
            else None
        )
        metadata_mtime = (
            os.stat(metadata_path).st_mtime_ns
            if os.path.exists(metadata_path)
            else None
        )
        return index_mtime, metadata_mtime

    def _load_shard(self, shard_id: int):
        index_path = os.path.join(self.source_dir, f"shard_{shard_id}.index")
        metadata_path = os.path.join(self.source_dir, f"shard_{shard_id}.metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            return None

        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = {str(k): v for k, v in json.load(f).items()}

        return FaissShard(shard_id=shard_id, index=index, metadata=metadata)

    def _load(self) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            if not os.path.isdir(self.source_dir):
                self._loaded = True
                return

            loaded_shards = []
            file_state = {}

            for shard_id in range(NUM_FAISS_SHARDS):
                state = self._get_file_state(shard_id)
                if state == (None, None):
                    continue

                try:
                    shard = self._load_shard(shard_id)
                except Exception as e:
                    print(
                        f"[FAISS Loader] {self.source} shard {shard_id} failed to load: {e}"
                    )
                    continue

                if shard is None:
                    continue

                loaded_shards.append(shard)
                file_state[shard_id] = state

            self.shards = loaded_shards
            self._file_state = file_state
            self._loaded = True

    def _needs_refresh(self) -> bool:
        if not self._loaded:
            return True

        if not os.path.isdir(self.source_dir):
            return bool(self.shards)

        current_state = {
            shard_id: self._get_file_state(shard_id)
            for shard_id in range(NUM_FAISS_SHARDS)
        }

        for shard_id in range(NUM_FAISS_SHARDS):
            previous = self._file_state.get(shard_id, (None, None))
            current = current_state[shard_id]
            if current != previous:
                return True

        return False

    def refresh(self) -> None:
        self._load()

        if not self._needs_refresh():
            return

        with self._lock:
            if not self._needs_refresh():
                return

            if not os.path.isdir(self.source_dir):
                self.shards = []
                self._file_state = {}
                return

            new_shards = []
            new_file_state = {}

            for shard_id in range(NUM_FAISS_SHARDS):
                state_before = self._get_file_state(shard_id)

                if state_before == (None, None):
                    continue

                try:
                    shard = self._load_shard(shard_id)
                except Exception as e:
                    print(
                        f"[FAISS Loader] {self.source} shard {shard_id} refresh failed: {e}"
                    )
                    continue

                if shard is None:
                    continue

                state_after = self._get_file_state(shard_id)

                if state_before != state_after:
                    continue

                new_shards.append(shard)
                new_file_state[shard_id] = state_after

            self.shards = new_shards
            self._file_state = new_file_state

    def load_async(self) -> None:
        threading.Thread(target=self._load, daemon=True).start()

    @property
    def total_shards(self) -> int:
        self._load()
        return len(self.shards)


class FaissRetriever:
    def __init__(self, source: str):
        self.source = source
        self.loader = FaissIndexLoader(source)
        self.loader.load_async()

    def _search_shard(
        self, shard: FaissShard, embedding: np.ndarray, top_k: int
    ) -> List[FaissSearchResult]:
        if shard.index.ntotal == 0:
            return []

        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        scores, ids = shard.index.search(query, top_k)
        results = []

        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id < 0:
                continue

            metadata = shard.metadata.get(str(int(faiss_id)))
            if metadata is None:
                continue

            vector_id = metadata.get("vector_id")
            document_id = metadata.get("document_id")
            source = metadata.get("source", self.source)
            embedding_type = metadata.get("embedding_type")

            if not vector_id or not document_id or not embedding_type:
                continue

            results.append(
                FaissSearchResult(
                    vector_id=vector_id,
                    document_id=document_id,
                    source=source,
                    embedding_type=embedding_type,
                    text=metadata.get("text", ""),
                    similarity=float(score),
                    chunk_id=metadata.get("chunk_id", -1),
                    start_char=metadata.get("start_char", 0),
                    end_char=metadata.get("end_char", 0),
                    sentence_count=metadata.get("sentence_count", 0),
                    average_similarity=metadata.get("average_similarity", 0.0),
                    token_estimate=metadata.get("token_estimate", 0),
                    metadata=metadata.get("metadata", {}),
                )
            )

        return results

    @staticmethod
    def _merge_results(results: List[FaissSearchResult]) -> List[FaissSearchResult]:
        merged: Dict[str, FaissSearchResult] = {}
        for result in results:
            existing = merged.get(result.vector_id)
            if existing is None or result.similarity > existing.similarity:
                merged[result.vector_id] = result

        return sorted(merged.values(), key=lambda result: result.similarity, reverse=True)

    def get_shards(self) -> List[FaissShard]:
        self.loader.refresh()
        return list(self.loader.shards)

    def search(self, embedding: np.ndarray, top_k: int) -> List[FaissSearchResult]:
        if top_k <= 0:
            return []

        shards = self.get_shards()
        if not shards:
            return []

        all_results = []
        futures = {
            FAISS_SEARCH_EXECUTOR.submit(self._search_shard, shard, embedding, top_k): shard.shard_id
            for shard in shards
        }

        for future in as_completed(futures):
            shard_id = futures[future]
            try:
                all_results.extend(future.result())
            except Exception as e:
                print(f"[FAISS Search] {self.source} shard {shard_id} failed: {e}")

        return self._merge_results(all_results)[:top_k]


class MultiSourceFaissRetriever:
    def __init__(self):
        self.retrievers: Dict[str, FaissRetriever] = {}
        self._lock = threading.Lock()

    def get_retriever(self, source: str) -> FaissRetriever:
        retriever = self.retrievers.get(source)
        if retriever is not None:
            return retriever

        with self._lock:
            retriever = self.retrievers.get(source)
            if retriever is None:
                retriever = FaissRetriever(source)
                self.retrievers[source] = retriever

        return retriever

    def search(
        self, embedding: np.ndarray, source_plans, top_k: int
    ) -> List[FaissSearchResult]:
        if top_k <= 0 or not source_plans:
            return []

        all_results = []
        future_map = {}

        for plan in source_plans:
            source = plan.source
            plan_top_k = max(1, int(plan.top_k))
            retriever = self.get_retriever(source)
            shards = retriever.get_shards()

            for shard in shards:
                future = FAISS_SEARCH_EXECUTOR.submit(
                    retriever._search_shard, shard, embedding, plan_top_k
                )
                future_map[future] = (source, shard.shard_id)

        for future in as_completed(future_map):
            source, shard_id = future_map[future]
            try:
                all_results.extend(future.result())
            except Exception as e:
                print(
                    f"[FAISS Search] Source '{source}' shard {shard_id} failed: {e}"
                )

        merged: Dict[str, FaissSearchResult] = {}
        for result in all_results:
            existing = merged.get(result.vector_id)
            if existing is None or result.similarity > existing.similarity:
                merged[result.vector_id] = result

        return sorted(
            merged.values(), key=lambda result: result.similarity, reverse=True
        )[:top_k]