# services/api/app/synax_embedding_generation.py
import hashlib
import json
import os
import numpy as np
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List

from services.api.app.synax_config import (
    embedder,
    VECTOR_DIM,
    EMBEDDING_OUTPUT_DIR,
    COREF_LINKED_ENTITY_OUTPUT_DIR,
)
from services.api.app.synax_entity_extraction import normalize_text
from services.api.app.synax_faiss_index import FaissIndexManager


@dataclass(slots=True)
class EmbeddingMetadata:
    document_id: str
    source: str
    embedding_type: str
    text: str = ""
    chunk_id: int = -1
    start_char: int = 0
    end_char: int = 0
    sentence_count: int = 0
    average_similarity: float = 0.0
    token_estimate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "embedding_type": self.embedding_type,
            "text": self.text,
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "sentence_count": self.sentence_count,
            "average_similarity": self.average_similarity,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EmbeddingRecord:
    vector_id: str
    embedding: List[float]
    metadata: EmbeddingMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "embedding": self.embedding,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(slots=True)
class EmbeddingJob:
    vector_id: str
    text: str
    filename: str
    metadata: EmbeddingMetadata


def make_vector_id(
    document_id: str, embedding_type: str, text: str, chunk_id: int = -1
) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    key = f"{document_id}:{embedding_type}:{chunk_id}:{text_hash}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class SemanticChunker:
    def __init__(
        self,
        embedder,
        minimum_similarity: float = 0.65,
        percentile: float = 20.0,
        max_characters: int = 1600,
        min_characters: int = 250,
    ):
        self.embedder = embedder
        self.minimum_similarity = minimum_similarity
        self.percentile = percentile
        self.max_characters = max_characters
        self.min_characters = min_characters

    def split_sentences(self, sentences: List[Dict[str, Any]]):
        return [
            {"text": s["text"], "start": s["start"], "end": s["end"]}
            for s in sentences
        ]

    def split(self, text: str, sentences: List[Dict[str, Any]]):
        sentences = self.split_sentences(sentences)
        if not sentences:
            return []

        sentence_texts = [s["text"] for s in sentences]
        vectors = self.embedder.encode(
            sentence_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        adjacent_similarities = [
            float(np.dot(vectors[i], vectors[i + 1]))
            for i in range(len(vectors) - 1)
        ]

        threshold = (
            max(self.minimum_similarity, np.percentile(adjacent_similarities, self.percentile))
            if adjacent_similarities
            else self.minimum_similarity
        )

        chunks = []
        chunk_sentences = [sentences[0]]
        chunk_vectors = [vectors[0]]
        similarities = []
        chunk_id = 0

        for sentence, vector in zip(sentences[1:], vectors[1:]):
            chunk_text = " ".join(s["text"] for s in chunk_sentences)
            candidate_text = " ".join(
                s["text"] for s in chunk_sentences + [sentence]
            )

            centroid = np.mean(chunk_vectors, axis=0)
            norm = np.linalg.norm(centroid)
            centroid = centroid / norm if norm > 1e-12 else np.zeros_like(centroid)
            similarity = float(np.dot(centroid, vector))

            exceeds_maximum = len(candidate_text) > self.max_characters
            semantic_boundary = (
                similarity < threshold and len(chunk_text) >= self.min_characters
            )
            should_split = exceeds_maximum or semantic_boundary

            if should_split and chunk_sentences:
                score = (
                    float(np.mean(similarities)) if similarities else 1.0
                )
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "start": chunk_sentences[0]["start"],
                        "end": chunk_sentences[-1]["end"],
                        "sentence_count": len(chunk_sentences),
                        "average_similarity": score,
                        "token_estimate": len(chunk_text.split()),
                    }
                )
                chunk_id += 1
                chunk_sentences = [sentence]
                chunk_vectors = [vector]
                similarities = []
            else:
                chunk_sentences.append(sentence)
                chunk_vectors.append(vector)
                similarities.append(similarity)

        if chunk_sentences:
            chunk_text = " ".join(s["text"] for s in chunk_sentences)
            score = float(np.mean(similarities)) if similarities else 1.0
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "start": chunk_sentences[0]["start"],
                    "end": chunk_sentences[-1]["end"],
                    "sentence_count": len(chunk_sentences),
                    "average_similarity": score,
                    "token_estimate": len(chunk_text.split()),
                }
            )

        return chunks


class EmbeddingGenerator:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunker = SemanticChunker(
            embedder=self.embedder,
            minimum_similarity=0.65,
            percentile=20,
            max_characters=1600,
            min_characters=250,
        )

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, VECTOR_DIM), dtype=np.float32)
        vectors = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def build_document_metadata(self, coref_document: Dict[str, Any]) -> Dict[str, Any]:
        upstream_metadata = dict(coref_document.get("metadata", {}))
        return upstream_metadata

    def build_metadata_embedding_text(self, coref_document: Dict[str, Any]) -> str:
        sections = []
        for cluster in coref_document.get("clusters", []):
            canonical = cluster.get("canonical_name")
            if not canonical:
                continue
            aliases = sorted(
                {
                    alias.strip()
                    for alias in cluster.get("aliases", [])
                    if isinstance(alias, str) and alias.strip()
                }
            )
            lines = [f"Entity: {canonical}"]
            if aliases:
                lines.append(f"Aliases: {', '.join(aliases)}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)


class EmbeddingIndex:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = self._load()

    def _load(self) -> Dict[str, str]:
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        directory = os.path.dirname(self.index_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    @staticmethod
    def compute_hash(rewritten_text: str, metadata_text: str) -> str:
        payload = json.dumps(
            {
                "rewritten_text": normalize_text(rewritten_text),
                "metadata_text": normalize_text(metadata_text),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def needs_update(
        self, document_id: str, rewritten_text: str, metadata_text: str
    ) -> bool:
        current_hash = self.compute_hash(
            rewritten_text=rewritten_text, metadata_text=metadata_text
        )
        return self.index.get(document_id) != current_hash

    def update(
        self, document_id: str, rewritten_text: str, metadata_text: str
    ) -> None:
        self.index[document_id] = self.compute_hash(
            rewritten_text=rewritten_text, metadata_text=metadata_text
        )


class EmbeddingGenerationPipeline:
    def __init__(self, embedder):
        self.generator = EmbeddingGenerator(embedder)
        self.batch_size = 256
        self.flush_threshold = 4096
        self.embedding_jobs = []
        self.pending_records = {}
        self.documents_pending_commit = {}
        self.embedding_index = EmbeddingIndex(
            os.path.join(EMBEDDING_OUTPUT_DIR, "embedding_index.json")
        )
        self.faiss_managers = {}
        self.pending_document_hashes = {}

    def document_hash(self, rewritten_text: str, metadata_text: str) -> str:
        return self.embedding_index.compute_hash(
            rewritten_text=rewritten_text, metadata_text=metadata_text
        )

    def load_coreference_document(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_faiss_manager(self, source: str) -> FaissIndexManager:
        manager = self.faiss_managers.get(source)
        if manager is None:
            manager = FaissIndexManager(source)
            self.faiss_managers[source] = manager
        return manager

    def save_embeddings(
        self, domain: str, filename: str, embeddings: List[EmbeddingRecord]
    ) -> None:
        domain_dir = os.path.join(EMBEDDING_OUTPUT_DIR, domain)
        os.makedirs(domain_dir, exist_ok=True)
        output_path = os.path.join(domain_dir, filename + ".embeddings.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [embedding.to_dict() for embedding in embeddings],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def process_embedding_jobs(self) -> None:
        if not self.embedding_jobs:
            return

        all_vectors = []
        for start in range(0, len(self.embedding_jobs), self.batch_size):
            batch = self.embedding_jobs[start : start + self.batch_size]
            texts = [job.text for job in batch]
            vectors = self.generator.generate_embeddings(texts)

            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(batch)}, got {len(vectors)}."
                )

            for job, vector in zip(batch, vectors):
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                if vector.shape[0] != VECTOR_DIM:
                    raise ValueError(
                        f"Embedding dimension mismatch for vector '{job.vector_id}': "
                        f"expected {VECTOR_DIM}, got {vector.shape[0]}."
                    )
                all_vectors.append((job, vector))

        documents_by_source = {}
        for job, _ in all_vectors:
            documents_by_source.setdefault(job.metadata.source, set()).add(
                job.metadata.document_id
            )

        for source, document_ids in documents_by_source.items():
            manager = self.get_faiss_manager(source)
            for document_id in document_ids:
                manager.remove_document(document_id)

        for job, vector in all_vectors:
            record = EmbeddingRecord(
                vector_id=job.vector_id,
                embedding=vector.tolist(),
                metadata=job.metadata,
            )
            document_id = job.metadata.document_id

            self.pending_records.setdefault(
                document_id, {"filename": job.filename, "records": []}
            )["records"].append(record)

            manager = self.get_faiss_manager(job.metadata.source)
            manager.insert_embedding(
                vector_id=record.vector_id,
                embedding=record.embedding,
                metadata=record.metadata.to_dict(),
            )

        self.embedding_jobs.clear()

    def flush_pending_records(self) -> int:
        total = 0
        for document_id, data in self.pending_records.items():
            filename = data["filename"]
            records = data["records"]
            if not records:
                continue

            source = records[0].metadata.source
            self.save_embeddings(domain=source, filename=filename, embeddings=records)
            total += len(records)

            pending_state = self.documents_pending_commit.get(document_id)
            if pending_state:
                rewritten_text, metadata_text = pending_state
                self.embedding_index.update(
                    document_id=document_id,
                    rewritten_text=rewritten_text,
                    metadata_text=metadata_text,
                )

        self.pending_records.clear()
        self.documents_pending_commit.clear()
        return total

    def flush_if_needed(self) -> int:
        if len(self.embedding_jobs) < self.flush_threshold:
            return 0
        self.process_embedding_jobs()
        return self.flush_pending_records()

    def build_document_jobs(
        self,
        coref_document: Dict[str, Any],
        rewritten_text: str,
        metadata_text: str,
        document_id: str,
        source: str,
        filename: str,
    ) -> List[EmbeddingJob]:
        jobs = []
        jobs.append(
            EmbeddingJob(
                vector_id=make_vector_id(
                    document_id=document_id,
                    embedding_type="document",
                    text=rewritten_text,
                ),
                filename=filename,
                text=rewritten_text,
                metadata=EmbeddingMetadata(
                    document_id=document_id,
                    source=source,
                    embedding_type="document",
                    text=rewritten_text,
                    sentence_count=len(coref_document.get("sentences", [])),
                    metadata=self.generator.build_document_metadata(coref_document),
                ),
            )
        )

        if metadata_text.strip():
            jobs.append(
                EmbeddingJob(
                    vector_id=make_vector_id(
                        document_id=document_id,
                        embedding_type="metadata",
                        text=metadata_text,
                    ),
                    filename=filename,
                    text=metadata_text,
                    metadata=EmbeddingMetadata(
                        document_id=document_id,
                        source=source,
                        embedding_type="metadata",
                        text=metadata_text,
                        sentence_count=len(coref_document.get("sentences", [])),
                        metadata=self.generator.build_document_metadata(coref_document),
                    ),
                )
            )

        return jobs

    def build_chunk_jobs(
        self,
        coref_document: Dict[str, Any],
        rewritten_text: str,
        document_id: str,
        source: str,
        filename: str,
    ) -> List[EmbeddingJob]:
        chunks = self.generator.chunker.split(
            rewritten_text, coref_document.get("sentences")
        )
        if not chunks:
            return []

        jobs = []
        for chunk in chunks:
            jobs.append(
                EmbeddingJob(
                    vector_id=make_vector_id(
                        document_id=document_id,
                        embedding_type="chunk",
                        text=chunk["text"],
                        chunk_id=chunk["chunk_id"],
                    ),
                    filename=filename,
                    text=chunk["text"],
                    metadata=EmbeddingMetadata(
                        document_id=document_id,
                        source=source,
                        embedding_type="chunk",
                        text=chunk["text"],
                        chunk_id=chunk["chunk_id"],
                        start_char=chunk["start"],
                        end_char=chunk["end"],
                        sentence_count=chunk["sentence_count"],
                        average_similarity=chunk["average_similarity"],
                        token_estimate=chunk["token_estimate"],
                    ),
                )
            )

        return jobs

    def process_document(self, filepath: str, domain: str) -> int:
        coref_document = self.load_coreference_document(filepath)
        rewritten_text = coref_document.get("rewritten_text", "").strip()

        if not rewritten_text:
            return 0

        filename = os.path.basename(filepath).replace(".coreflinked.json", "")
        document_id = coref_document.get("document_id")
        mentions = coref_document.get("mentions", [])
        source = mentions[0].get("source", "") if mentions else ""

        if not document_id:
            raise ValueError(f"{domain}/{filename} is missing document_id.")

        metadata_text = self.generator.build_metadata_embedding_text(coref_document)
        current_hash = self.document_hash(
            rewritten_text=rewritten_text, metadata_text=metadata_text
        )
        pending_hash = self.pending_document_hashes.get(document_id)

        if pending_hash == current_hash:
            print(f"[Embedding Generation] {domain}/{filename} → skipped")
            return 0

        if (
            pending_hash is None
            and not self.embedding_index.needs_update(
                document_id=document_id,
                rewritten_text=rewritten_text,
                metadata_text=metadata_text,
            )
        ):
            print(f"[Embedding Generation] {domain}/{filename} → skipped")
            return 0

        jobs = self.build_document_jobs(
            coref_document=coref_document,
            rewritten_text=rewritten_text,
            metadata_text=metadata_text,
            document_id=document_id,
            source=source,
            filename=filename,
        )
        jobs.extend(
            self.build_chunk_jobs(
                coref_document=coref_document,
                rewritten_text=rewritten_text,
                document_id=document_id,
                source=source,
                filename=filename,
            )
        )

        self.embedding_jobs.extend(jobs)
        self.documents_pending_commit[document_id] = (rewritten_text, metadata_text)
        self.pending_document_hashes[document_id] = current_hash
        written = self.flush_if_needed()

        return written

    def process_dataset(self) -> int:
        total_domains = 0
        total_documents = 0
        total_embeddings = 0

        for domain in os.listdir(COREF_LINKED_ENTITY_OUTPUT_DIR):
            domain_dir = os.path.join(COREF_LINKED_ENTITY_OUTPUT_DIR, domain)
            if not os.path.isdir(domain_dir):
                continue

            print(f"\n{domain.upper()} EMBEDDING GENERATION")
            total_domains += 1

            for file in os.listdir(domain_dir):
                if not file.endswith(".coreflinked.json"):
                    continue

                filepath = os.path.join(domain_dir, file)
                try:
                    total_documents += 1
                    total_embeddings += self.process_document(
                        filepath=filepath, domain=domain
                    )
                except Exception as e:
                    print("[Embedding Generation Error]", e)

        self.process_embedding_jobs()
        total_embeddings += self.flush_pending_records()

        for manager in self.faiss_managers.values():
            manager.save_all()

        self.embedding_index.save()
        self.pending_document_hashes.clear()

        print("\nEMBEDDING GENERATION COMPLETE")
        print(f"Domains Processed: {total_domains}")
        print(f"Documents Processed: {total_documents}")
        print(f"Embeddings Generated: {total_embeddings}")

        return total_embeddings


def run_embedding_generation():
    return EmbeddingGenerationPipeline(embedder).process_dataset()