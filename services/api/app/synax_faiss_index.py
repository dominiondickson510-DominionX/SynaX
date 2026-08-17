# services/api/app/synax_faiss_index.py
import os
import json
import hashlib
import faiss
import numpy as np
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List

from services.api.app.synax_config import (
    FAISS_SHARD_DIR,
    VECTOR_DIM,
    NUM_FAISS_SHARDS,
)


def vector_id_to_int64(vector_id: str) -> np.int64:
    digest = hashlib.sha256(vector_id.encode("utf-8")).digest()
    return np.int64(int.from_bytes(digest[:8], byteorder="big", signed=True))


@dataclass(slots=True)
class VectorRecord:
    vector_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]):
        return cls(
            vector_id=obj["vector_id"],
            embedding=np.asarray(obj["embedding"], dtype=np.float32),
            metadata=obj["metadata"],
        )


class FaissIndexTracker:
    def __init__(self, source: str):
        self.source = source
        self.source_dir = os.path.join(FAISS_SHARD_DIR, source)
        os.makedirs(self.source_dir, exist_ok=True)
        self.index_path = os.path.join(self.source_dir, "faiss_index.json")
        self.index = self._load()

    def _load(self) -> Dict[str, bool]:
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def needs_indexing(self, vector_id: str) -> bool:
        return vector_id not in self.index

    def mark_indexed(self, vector_id: str) -> None:
        self.index[vector_id] = True

    def remove(self, vector_id: str) -> None:
        self.index.pop(vector_id, None)

    def save(self) -> None:
        directory = os.path.dirname(self.index_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)


class FaissShard:
    def __init__(self, source: str, shard_id: int):
        self.source = source
        self.shard_id = shard_id
        self.source_dir = os.path.join(FAISS_SHARD_DIR, source)
        os.makedirs(self.source_dir, exist_ok=True)
        self.index_path = os.path.join(self.source_dir, f"shard_{shard_id}.index")
        self.metadata_path = os.path.join(
            self.source_dir, f"shard_{shard_id}.metadata.json"
        )

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(VECTOR_DIM))

    def add(self, record: VectorRecord) -> bool:
        embedding = record.embedding.astype(np.float32, copy=False).reshape(-1)
        if embedding.shape[0] != VECTOR_DIM:
            raise ValueError(
                f"Embedding dimension mismatch for vector '{record.vector_id}': "
                f"expected {VECTOR_DIM}, got {embedding.shape[0]}.",
            )

        faiss_id = vector_id_to_int64(record.vector_id)
        metadata_key = str(int(faiss_id))

        if metadata_key in self.metadata:
            return False

        self.index.add_with_ids(
            embedding.reshape(1, -1), np.asarray([faiss_id], dtype=np.int64)
        )
        self.metadata[metadata_key] = {
            "vector_id": record.vector_id,
            **record.metadata,
        }
        return True

    def remove_document(self, document_id: str) -> List[str]:
        vector_ids = [
            metadata["vector_id"]
            for metadata in self.metadata.values()
            if metadata.get("document_id") == document_id
        ]
        if not vector_ids:
            return []

        self.index.remove_ids(
            np.asarray(
                [vector_id_to_int64(vector_id) for vector_id in vector_ids],
                dtype=np.int64,
            )
        )
        for vector_id in vector_ids:
            self.metadata.pop(str(int(vector_id_to_int64(vector_id))), None)

        return vector_ids

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)


class FaissIndexManager:
    def __init__(self, source: str):
        self.source = source
        self.source_dir = os.path.join(FAISS_SHARD_DIR, source)
        os.makedirs(self.source_dir, exist_ok=True)
        self.shards = {
            shard: FaissShard(source, shard) for shard in range(NUM_FAISS_SHARDS)
        }
        self.tracker = FaissIndexTracker(source)

    def determine_shard(self, vector_id: str) -> int:
        digest = hashlib.sha256(vector_id.encode("utf-8")).hexdigest()
        return int(digest, 16) % NUM_FAISS_SHARDS

    def insert(self, record: VectorRecord) -> bool:
        if not self.tracker.needs_indexing(record.vector_id):
            return False

        shard = self.determine_shard(record.vector_id)
        inserted = self.shards[shard].add(record)

        if not inserted:
            return False

        self.tracker.mark_indexed(record.vector_id)
        return True

    def insert_embedding(
        self, vector_id: str, embedding: List[float], metadata: Dict[str, Any]
    ) -> bool:
        return self.insert(
            VectorRecord(
                vector_id=vector_id,
                embedding=np.asarray(embedding, dtype=np.float32),
                metadata=metadata,
            )
        )

    def remove_document(self, document_id: str) -> int:
        removed_vector_ids = []
        for shard in self.shards.values():
            removed_vector_ids.extend(shard.remove_document(document_id))

        for vector_id in removed_vector_ids:
            self.tracker.remove(vector_id)

        return len(removed_vector_ids)

    def save_all(self):
        for shard in self.shards.values():
            shard.save()
        self.tracker.save()