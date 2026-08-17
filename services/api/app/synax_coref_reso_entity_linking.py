# services/api/app/synax_coref_reso_entity_linking.py
import hashlib
import json
import os
import unicodedata
import faiss
import numpy as np
import spacy
import fastcoref
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from services.api.app.synax_config import (
    DATA_DIR,
    ENTITY_OUTPUT_DIR,
    COREF_LINKED_ENTITY_OUTPUT_DIR,
    VECTOR_DIM,
    EMBEDDING_SIMILARITY_THRESHOLD,
    embedder,
    reranker,
    reranker_tokenizer,
)
from services.api.app.synax_entity_extraction import (
    EntityType,
    normalize_text,
)


@dataclass(slots=True)
class EntityCandidate:
    entity: Dict[str, Any]
    score: float
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntityMatchResult:
    entity: Optional[Dict[str, Any]]
    confidence: float
    method: str
    candidates: List[EntityCandidate] = field(default_factory=list)


class EntitySearchIndex:
    def __init__(self):
        self.entities: List[Dict[str, Any]] = []
        self.faiss_indices: Dict = {
            entity_type: faiss.IndexFlatIP(VECTOR_DIM)
            for entity_type in EntityType
        }
        self.embedding_entities: Dict = {
            entity_type: [] for entity_type in EntityType
        }
        self.indexed_entity_ids: Set[str] = set()
        self.extracted_entities_lookup: Dict[str, Dict[str, Any]] = {}


@dataclass(slots=True)
class SentenceSpan:
    index: int
    text: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }


@dataclass(slots=True)
class ResolvedMention:
    text: str
    start: int
    end: int
    sentence_index: int
    cluster_id: str
    entity_id: str
    canonical_name: str
    entity_type: str
    source: str
    document_id: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "sentence_index": self.sentence_index,
            "cluster_id": self.cluster_id,
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "source": self.source,
            "document_id": self.document_id,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class CoreferenceMention:
    text: str
    start: int
    end: int
    sentence_index: int
    entity: Optional[Dict[str, Any]] = None
    is_pronoun: bool = False
    is_possessive: bool = False
    is_reflexive: bool = False
    is_demonstrative: bool = False
    is_relative: bool = False

    @property
    def entity_id(self) -> Optional[str]:
        return None if self.entity is None else self.entity.get("entity_id")

    @property
    def canonical_name(self) -> Optional[str]:
        return None if self.entity is None else self.entity.get("canonical_name")

    @property
    def entity_type(self) -> Optional[str]:
        return None if self.entity is None else self.entity.get("entity_type")

    @property
    def grammatical_role(self) -> str:
        if self.is_possessive:
            return "possessive"
        if self.is_reflexive:
            return "reflexive"
        if self.is_demonstrative:
            return "demonstrative"
        if self.is_relative:
            return "relative"
        if self.is_pronoun:
            return "pronoun"
        return "nominal"

    @property
    def should_rewrite(self) -> bool:
        return not (self.is_relative or self.is_reflexive)


@dataclass(slots=True)
class CoreferenceCluster:
    cluster_id: str
    canonical_entity: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    mentions: List[CoreferenceMention] = field(default_factory=list)

    @property
    def entity_id(self) -> Optional[str]:
        return (
            None
            if self.canonical_entity is None
            else self.canonical_entity.get("entity_id")
        )

    @property
    def canonical_name(self) -> Optional[str]:
        return (
            None
            if self.canonical_entity is None
            else self.canonical_entity.get("canonical_name")
        )

    @property
    def entity_type(self) -> Optional[str]:
        return (
            None
            if self.canonical_entity is None
            else self.canonical_entity.get("entity_type")
        )

    def add(self, mention: CoreferenceMention) -> None:
        self.mentions.append(mention)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "aliases": (
                self.canonical_entity.get("aliases", [])
                if self.canonical_entity
                else []
            ),
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "mentions": [
                {
                    "text": m.text,
                    "start": m.start,
                    "end": m.end,
                    "sentence_index": m.sentence_index,
                }
                for m in self.mentions
            ],
        }


@dataclass(slots=True)
class RewriteOperation:
    start: int
    end: int
    original: str
    replacement: str
    cluster_id: str
    entity_id: str
    canonical_name: str
    grammatical_role: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "original": self.original,
            "replacement": self.replacement,
            "cluster_id": self.cluster_id,
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "grammatical_role": self.grammatical_role,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class CoreferenceDocument:
    text: str
    rewritten_text: str
    document_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sentences: List[SentenceSpan] = field(default_factory=list)
    mentions: List[ResolvedMention] = field(default_factory=list)
    clusters: List[CoreferenceCluster] = field(default_factory=list)
    rewrite_operations: List[RewriteOperation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "metadata": self.metadata,
            "rewritten_text": self.rewritten_text,
            "sentences": [s.to_dict() for s in self.sentences],
            "mentions": [m.to_dict() for m in self.mentions],
            "clusters": [c.to_dict() for c in self.clusters],
            "rewrite_operations": [r.to_dict() for r in self.rewrite_operations],
        }


class BaseCoreferenceResolver:
    name = "base"

    def resolve(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> CoreferenceDocument:
        raise NotImplementedError


class CorefResoEntityLinker(BaseCoreferenceResolver):
    name = "fastcoref"

    def __init__(self, language: str = "en", model_architecture: str = "LingMessCoref"):
        self.nlp = spacy.load("en_core_web_trf")
        if "fastcoref" not in self.nlp.pipe_names:
            self.nlp.add_pipe(
                "fastcoref",
                config={
                    "model_architecture": model_architecture,
                    "device": "cuda" if spacy.prefer_gpu() else "cpu",
                },
            )
        self.search_index: Optional[EntitySearchIndex] = None
        self.embedder = embedder
        self.reranker = reranker
        self.reranker_tokenizer = reranker_tokenizer
        self.top_embedding_candidates = 20
        self.top_rerank_candidates = 5
        self.embedding_threshold = EMBEDDING_SIMILARITY_THRESHOLD

    def _normalize_name(self, text: str) -> str:
        text = normalize_text(text).casefold()
        text = text.replace("'", "'").replace("'", "'").replace("`", "'")
        text = text.replace("–", "-").replace("—", "-").replace("−", "-")
        return text

    def _candidate_strings(self, entity: Dict[str, Any]) -> List[str]:
        strings: Set[str] = set()
        canonical = entity.get("canonical_name")
        if isinstance(canonical, str):
            canonical = self._normalize_name(canonical)
            if canonical:
                strings.add(canonical)
        aliases = entity.get("aliases") or []
        for alias in aliases:
            if not isinstance(alias, str):
                continue
            alias = self._normalize_name(alias)
            if alias:
                strings.add(alias)
        return sorted(strings)

    def _entity_representation(self, entity: Dict[str, Any]) -> str:
        parts = []
        canonical = (entity.get("canonical_name") or "").strip()
        if canonical:
            parts.append(f"Name:{canonical}")
        aliases = [
            a.strip()
            for a in (entity.get("aliases") or [])
            if isinstance(a, str) and a.strip()
        ]
        if aliases:
            parts.append("Aliases:" + ",".join(aliases))
        entity_type = (entity.get("entity_type") or "").strip()
        if entity_type:
            parts.append(f"Type:{entity_type}")
        metadata = entity.get("metadata") or {}
        descriptions = metadata.get("descriptions")
        if isinstance(descriptions, str):
            if descriptions.strip():
                parts.append(f"Description:{descriptions.strip()}")
        elif isinstance(descriptions, dict):
            description = descriptions.get("en") or next(
                (
                    v
                    for v in descriptions.values()
                    if isinstance(v, str) and v.strip()
                ),
                None,
            )
            if isinstance(description, str):
                parts.append(f"Description:{description.strip()}")
        labels = metadata.get("labels")
        if isinstance(labels, dict):
            label = labels.get("en") or next(
                (v for v in labels.values() if isinstance(v, str) and v.strip()),
                None,
            )
            if isinstance(label, str):
                parts.append(f"Label:{label.strip()}")
        elif isinstance(labels, str):
            if labels.strip():
                parts.append(f"Label:{labels.strip()}")
        for field_name in ["title", "abstract", "description"]:
            value = metadata.get(field_name)
            if isinstance(value, str) and value.strip():
                parts.append(f"{field_name.capitalize()}:{value.strip()}")
        for field_name in ["authors", "institutions", "concepts", "mesh_terms"]:
            values = metadata.get(field_name)
            if isinstance(values, list) and values:
                cleaned = [str(v).strip() for v in values if str(v).strip()]
                if cleaned:
                    parts.append(f"{field_name.capitalize()}:{','.join(cleaned)}")
        return "\n".join(parts)

    def _build_entity_index(self, entities: List[Dict[str, Any]]) -> None:
        if not entities:
            return
        grouped_entities = {}
        for entity in entities:
            entity_id = entity.get("entity_id")
            if not entity_id or entity_id in self.search_index.indexed_entity_ids:
                continue
            self.search_index.indexed_entity_ids.add(entity_id)
            self.search_index.entities.append(entity)
            strings = self._candidate_strings(entity)
            for string in strings:
                self.search_index.extracted_entities_lookup[string] = entity
            entity_type = EntityType(entity["entity_type"])
            grouped_entities.setdefault(entity_type, []).append(entity)
        for entity_type, group in grouped_entities.items():
            texts = [self._entity_representation(entity) for entity in group]
            vectors = self.embedder.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            self.search_index.faiss_indices[entity_type].add(vectors)
            self.search_index.embedding_entities[entity_type].extend(group)

    def _map_sentence_index(self, char_start: int, sentence_spans) -> int:
        for index, sentence in enumerate(sentence_spans):
            if sentence.start <= char_start < sentence.end:
                return index
        return 0

    def _cluster_context(self, text: str, sentence_spans, cluster) -> str:
        selected = []
        seen = set()
        for mention in cluster:
            sentence_index = self._map_sentence_index(mention.start, sentence_spans)
            for index in (sentence_index - 1, sentence_index, sentence_index + 1):
                if index < 0 or index >= len(sentence_spans):
                    continue
                sentence = sentence_spans[index]
                key = (sentence.start, sentence.end)
                if key in seen:
                    continue
                seen.add(key)
                selected.append(text[sentence.start : sentence.end].strip())
        return "\n".join(selected)

    def _encode_cluster(self, text, sentence_spans, cluster) -> np.ndarray:
        context = self._cluster_context(text, sentence_spans, cluster)
        embedding = self.embedder.encode(
            context,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)

    def _contextual_embedding_similarity(
        self, text: str, sentence_spans, cluster, entity_type: str
    ) -> List[EntityCandidate]:
        entity_type = EntityType(entity_type)
        index = self.search_index.faiss_indices[entity_type]
        if index.ntotal == 0:
            return []
        cluster_embedding = self._encode_cluster(text, sentence_spans, cluster)
        scores, indices = index.search(
            cluster_embedding.reshape(1, -1), self.top_embedding_candidates
        )
        ranked = []
        for similarity, idx in zip(scores[0], indices[0]):
            if idx == -1 or similarity < self.embedding_threshold:
                continue
            entity = self.search_index.embedding_entities[entity_type][idx]
            ranked.append(
                EntityCandidate(
                    entity=entity,
                    score=float(similarity),
                    method="contextual_embedding",
                    metadata={"cosine_similarity": float(similarity)},
                )
            )
        return ranked[: self.top_rerank_candidates]

    def _cross_encoder_scores(
        self, context: str, candidates: List[EntityCandidate]
    ) -> List[EntityCandidate]:
        if not candidates:
            return []
        pairs = [
            (context, self._entity_representation(candidate.entity))
            for candidate in candidates
        ]
        encoded = self.reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024,
        )
        device = next(self.reranker.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.reranker(**encoded)
        logits = outputs.logits.view(-1).float()
        confidences = torch.sigmoid(logits).cpu().numpy()
        logits = logits.cpu().numpy()
        ranked = []
        for candidate, logit, confidence in zip(candidates, logits, confidences):
            ranked.append(
                EntityCandidate(
                    entity=candidate.entity,
                    score=float(logit),
                    method="cross_encoder",
                    metadata={
                        "reranker_logit": float(logit),
                        "reranker_confidence": float(confidence),
                        "embedding_score": candidate.score,
                    },
                )
            )
        ranked.sort(key=lambda candidate: candidate.score, reverse=True)
        return ranked

    def _match_cluster_entity(
        self, text, sentence_spans, cluster, entity_type: str
    ) -> Optional[EntityCandidate]:
        contextual_candidates = self._contextual_embedding_similarity(
            text, sentence_spans, cluster, entity_type
        )
        if not contextual_candidates:
            return None
        reranked = self._cross_encoder_scores(
            self._cluster_context(text, sentence_spans, cluster),
            contextual_candidates,
        )
        if not reranked:
            return None
        return reranked[0]

    def _find_cluster_anchor(self, cluster) -> Optional[EntityCandidate]:
        candidates: List[EntityCandidate] = []
        for mention in cluster:
            normalized = self._normalize_name(mention.text)
            entity = self.search_index.extracted_entities_lookup.get(normalized)
            if entity is None:
                continue
            canonical = self._normalize_name(entity.get("canonical_name") or "")
            if normalized == canonical:
                score = 100.0
                match_type = "canonical"
            else:
                aliases = {
                    self._normalize_name(alias)
                    for alias in (entity.get("aliases") or [])
                    if isinstance(alias, str)
                }
                if normalized in aliases:
                    score = 75.0
                    match_type = "alias"
                else:
                    continue
            score += min(len(normalized), 30)
            score += max(0, 20 - mention.start / 1000.0)
            candidates.append(
                EntityCandidate(
                    entity=entity,
                    score=score,
                    method="cluster_anchor",
                    metadata={
                        "match_type": match_type,
                        "mention": mention.text,
                        "normalized_mention": normalized,
                        "mention_length": len(normalized),
                        "start": mention.start,
                        "end": mention.end,
                    },
                )
            )
        if not candidates:
            return None
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[0]

    def _classify_mention(self, doc, start: int, end: int) -> Dict[str, bool]:
        span = doc.char_span(start, end, alignment_mode="expand")
        if span is None or len(span) == 0:
            return {
                "is_pronoun": False,
                "is_possessive": False,
                "is_reflexive": False,
                "is_demonstrative": False,
                "is_relative": False,
            }
        token = span.root
        morph = token.morph
        lower = token.lower_
        pronoun = token.pos_ == "PRON" or token.tag_ in {"PRP", "PRP$", "WP", "WP$", "DT"}
        possessive = token.tag_ in {"PRP$", "WP$"} or "Poss=Yes" in morph
        reflexive = "Reflex=Yes" in morph or lower.endswith("self") or lower.endswith("selves")
        demonstrative = lower in {"this", "that", "these", "those"}
        relative = token.tag_ in {"WP", "WP$"} or lower in {
            "who",
            "whom",
            "whose",
            "which",
            "that",
        }
        return {
            "is_pronoun": pronoun,
            "is_possessive": possessive,
            "is_reflexive": reflexive,
            "is_demonstrative": demonstrative,
            "is_relative": relative,
        }

    def _build_clusters(
        self, doc, text: str, sentence_spans
    ) -> List[CoreferenceCluster]:
        clusters: List[CoreferenceCluster] = []
        coref_clusters = getattr(doc._, "coref_clusters", None)
        if not coref_clusters:
            return clusters
        for cluster in coref_clusters:
            if (anchor := self._find_cluster_anchor(cluster)) is None:
                continue
            if (
                best_candidate := self._match_cluster_entity(
                    text=text,
                    sentence_spans=sentence_spans,
                    cluster=cluster,
                    entity_type=anchor.entity["entity_type"],
                )
            ) is None:
                continue
            entity = best_candidate.entity
            confidence = best_candidate.metadata.get(
                "reranker_confidence", best_candidate.score
            )
            cluster_id = hashlib.sha256(
                entity["entity_id"].encode("utf-8")
            ).hexdigest()[:16]
            output = CoreferenceCluster(
                cluster_id=cluster_id, canonical_entity=entity, confidence=confidence
            )
            for mention in cluster:
                start = mention.start
                end = mention.end
                if start >= end:
                    continue
                sentence_index = self._map_sentence_index(start, sentence_spans)
                flags = self._classify_mention(doc, start, end)
                output.add(
                    CoreferenceMention(
                        text=mention.text,
                        start=start,
                        end=end,
                        sentence_index=sentence_index,
                        entity=entity,
                        is_pronoun=flags["is_pronoun"],
                        is_possessive=flags["is_possessive"],
                        is_reflexive=flags["is_reflexive"],
                        is_demonstrative=flags["is_demonstrative"],
                        is_relative=flags["is_relative"],
                    )
                )
            if output.mentions:
                clusters.append(output)
        return clusters

    def _rewrite_document(
        self, text: str, clusters: List[CoreferenceCluster]
    ) -> tuple[str, List[RewriteOperation]]:
        replacements = []
        for cluster in clusters:
            canonical = cluster.canonical_name
            if not canonical:
                continue
            first_mention = True
            for mention in sorted(cluster.mentions, key=lambda m: m.start):
                if first_mention:
                    first_mention = False
                    continue
                if not mention.should_rewrite:
                    continue
                replacement = canonical
                if mention.is_possessive:
                    replacement = (
                        canonical
                        if canonical.endswith(("s", "S"))
                        else canonical + "'s"
                    )
                original = mention.text
                if original.isupper():
                    replacement = replacement.upper()
                elif original[:1].isupper() and original[1:].islower():
                    replacement = replacement[:1].upper() + replacement[1:]
                replacements.append(
                    RewriteOperation(
                        start=mention.start,
                        end=mention.end,
                        original=mention.text,
                        replacement=replacement,
                        cluster_id=cluster.cluster_id,
                        entity_id=cluster.entity_id,
                        canonical_name=cluster.canonical_name,
                        grammatical_role=mention.grammatical_role,
                        confidence=cluster.confidence,
                    )
                )
        replacements.sort(key=lambda operation: operation.start)
        rewritten = []
        cursor = 0
        for operation in replacements:
            if operation.start < cursor:
                continue
            rewritten.append(text[cursor : operation.start])
            rewritten.append(operation.replacement)
            cursor = operation.end
        rewritten.append(text[cursor :])
        return "".join(rewritten), replacements

    def resolve(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> CoreferenceDocument:
        self.search_index = EntitySearchIndex()
        doc = self.nlp(
            text, component_cfg={"fastcoref": {"resolve_text": False}}
        )
        sentence_spans = list(doc.sents)
        sentences = [
            SentenceSpan(
                index=i,
                text=text[s.start_char : s.end_char],
                start=s.start_char,
                end=s.end_char,
            )
            for i, s in enumerate(sentence_spans)
        ]
        self._build_entity_index(entities or [])
        clusters = self._build_clusters(doc=doc, text=text, sentence_spans=sentences)
        rewritten_text, rewrite_operations = self._rewrite_document(
            text=text, clusters=clusters
        )
        mentions = [
            ResolvedMention(
                text=mention.text,
                start=mention.start,
                end=mention.end,
                sentence_index=mention.sentence_index,
                cluster_id=cluster.cluster_id,
                entity_id=cluster.entity_id,
                canonical_name=cluster.canonical_name,
                entity_type=cluster.entity_type,
                source=cluster.canonical_entity.get("source", ""),
                document_id=cluster.canonical_entity["document_id"],
                confidence=cluster.confidence,
            )
            for cluster in clusters
            for mention in cluster.mentions
        ]
        document_id = mentions[0].document_id if mentions else ""
        return CoreferenceDocument(
            text=text,
            rewritten_text=rewritten_text,
            document_id=document_id,
            sentences=sentences,
            mentions=mentions,
            clusters=clusters,
            rewrite_operations=rewrite_operations,
        )


@dataclass(slots=True)
class CorefResoEntityLinkingStatistics:
    documents_processed: int = 0
    clusters_created: int = 0
    mentions_resolved: int = 0
    entities_matched: int = 0
    average_confidence: float = 0.0
    confidence_scores: List[float] = field(default_factory=list)
    domain_statistics: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(
            lambda: {
                "documents": 0,
                "clusters": 0,
                "mentions": 0,
                "matched_entities": 0,
            }
        )
    )

    def record_document(self, domain: str) -> None:
        self.documents_processed += 1
        self.domain_statistics[domain]["documents"] += 1

    def record_cluster(self, domain: str, cluster) -> None:
        self.clusters_created += 1
        self.domain_statistics[domain]["clusters"] += 1
        mention_count = len(cluster.mentions)
        self.mentions_resolved += mention_count
        self.domain_statistics[domain]["mentions"] += mention_count
        if cluster.canonical_entity is not None:
            self.entities_matched += 1
            self.domain_statistics[domain]["matched_entities"] += 1
        if cluster.confidence is not None:
            self.confidence_scores.append(float(cluster.confidence))

    def finalize(self) -> None:
        if self.confidence_scores:
            self.average_confidence = sum(self.confidence_scores) / len(
                self.confidence_scores
            )
        else:
            self.average_confidence = 0.0

    def print_domain_statistics(self) -> None:
        if not self.domain_statistics:
            return
        print("\nPER-DOMAIN STATISTICS")
        for domain in sorted(self.domain_statistics):
            stats = self.domain_statistics[domain]
            print(f"\n{domain.upper()}")
            print(f"Documents:{stats['documents']}")
            print(f"Clusters:{stats['clusters']}")
            print(f"Mentions:{stats['mentions']}")
            print(f"Matched Entities:{stats['matched_entities']}")


class CorefResoEntityLinkingPipeline:
    def __init__(self):
        self.resolver = CorefResoEntityLinker()
        self.statistics = CorefResoEntityLinkingStatistics()

    def compute_document_hash(self, text: str) -> str:
        return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()

    def needs_processing(
        self, domain: str, filename: str, document_hash: str
    ) -> bool:
        output = os.path.join(
            COREF_LINKED_ENTITY_OUTPUT_DIR, domain, filename + ".coreflinked.json"
        )
        if not os.path.exists(output):
            return True
        try:
            with open(output, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("document_hash") != document_hash
        except Exception:
            return True

    def load_document(self, filepath: str, domain: str) -> tuple[str, Dict[str, Any]]:
        def flatten_pmc_sections(sections):
            blocks = []

            def walk(section):
                if heading := section.get("heading", ""):
                    blocks.append(heading)
                for item in section.get("content", []):
                    if item_type := item.get("type"):
                        if item_type == "paragraph":
                            if text := item.get("text", ""):
                                blocks.append(text)
                        elif item_type == "table":
                            if caption := item.get("caption", ""):
                                blocks.append(caption)
                            for row in item.get("rows", []):
                                if isinstance(row, dict):
                                    blocks.append(
                                        " ".join(
                                            str(v) for v in row.values() if v
                                        )
                                    )
                                elif isinstance(row, list):
                                    blocks.append(
                                        " ".join(str(v) for v in row if v)
                                    )
                        elif item_type == "figure":
                            for caption in item.get("caption", []):
                                if caption:
                                    blocks.append(caption)
                        elif item_type in {
                            "display_equation",
                            "inline_equation",
                        }:
                            if formula := item.get("latex", ""):
                                blocks.append(formula)

                for child in section.get("children", []):
                    walk(child)

            for section in sections:
                walk(section)
            return "\n\n".join(blocks)

        metadata = {}
        text = ""
        if filepath.endswith(".txt"):
            with open(
                filepath, "r", encoding="utf-8", errors="ignore"
            ) as f:
                text = f.read()
            json_path = os.path.splitext(filepath)[0] + ".json"
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        if isinstance((obj := json.load(jf)), dict):
                            metadata = obj
                except Exception:
                    pass
            return text, metadata
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return str(obj), {}
        metadata = obj
        source = obj.get("source")
        if source in {"openalex", "pubmed"}:
            text = "\n".join(
                filter(
                    None,
                    [obj.get("title", ""), obj.get("abstract", "")],
                )
            )
        elif source == "arxiv":
            text = "\n".join(
                filter(
                    None,
                    [
                        obj.get("title", ""),
                        obj.get("abstract", ""),
                        obj.get("full_text", ""),
                    ],
                )
            )
        elif source == "wikidata":
            labels = obj.get("labels", {})
            descriptions = obj.get("descriptions", {})
            entity_claims = obj.get("entity_claims", {})
            blocks = [labels.get("en", ""), descriptions.get("en", "")]
            for property_name, values in sorted(entity_claims.items()):
                if values:
                    blocks.append(f"{property_name}: {', '.join(values)}")
            text = "\n\n".join(filter(None, blocks))
        elif source == "pubmedcentral":
            text = "\n\n".join(
                filter(
                    None,
                    [
                        obj.get("title", ""),
                        obj.get("abstract", ""),
                        flatten_pmc_sections(obj.get("sections", [])),
                        obj.get("acknowledgements", ""),
                        "\n".join(
                            ref.get("text", "")
                            for ref in obj.get("references", [])
                            if ref.get("text")
                        ),
                    ],
                )
            )
        else:
            text = json.dumps(obj, ensure_ascii=False)
        return text.strip(), metadata

    def load_extracted_entities(self, filepath: str) -> List[Dict[str, Any]]:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_result(
        self,
        domain: str,
        filename: str,
        result: CoreferenceDocument,
        document_hash: str,
    ):
        domain_dir = os.path.join(COREF_LINKED_ENTITY_OUTPUT_DIR, domain)
        os.makedirs(domain_dir, exist_ok=True)
        output = os.path.join(domain_dir, filename + ".coreflinked.json")
        payload = result.to_dict()
        payload["document_hash"] = document_hash
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def process_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        entity_path: str,
        domain: str,
        filename: str,
        document_hash: str,
    ):
        entities = self.load_extracted_entities(entity_path)
        result = self.resolver.resolve(text=text, entities=entities)
        result.metadata = metadata
        self.statistics.record_document(domain)
        for cluster in result.clusters:
            self.statistics.record_cluster(domain, cluster)
        self.save_result(domain, filename, result, document_hash)
        print(f"[Coreference Reso Entity Linking] {domain}/{filename} → {len(result.mentions)} mentions")
        return len(result.mentions)

    def process_dataset(self):
        total_domains = 0
        for domain in os.listdir(DATA_DIR):
            document_dir = os.path.join(DATA_DIR, domain)
            entity_dir = os.path.join(ENTITY_OUTPUT_DIR, domain)
            if not os.path.isdir(document_dir) or not os.path.isdir(entity_dir):
                continue
            print(f"\n{domain.upper()} COREF RESO ENTITY LINKING")
            for file in os.listdir(document_dir):
                if not file.endswith((".txt", ".json")):
                    continue
                text_path = os.path.join(document_dir, file)
                filename = os.path.splitext(file)[0]
                entity_path = os.path.join(entity_dir, filename + ".entities.json")
                if not os.path.exists(entity_path):
                    continue
                try:
                    text, metadata = self.load_document(
                        filepath=text_path, domain=domain
                    )
                    document_hash = self.compute_document_hash(text)
                    if not self.needs_processing(
                        domain=domain, filename=filename, document_hash=document_hash
                    ):
                        print(f"[Skipped] {domain}/{filename}")
                        continue
                    self.process_document(
                        text=text,
                        metadata=metadata,
                        entity_path=entity_path,
                        domain=domain,
                        filename=filename,
                        document_hash=document_hash,
                    )
                except Exception as e:
                    print("[Coreference Reso Entity Linking Error]", e)
            total_domains += 1
        self.statistics.finalize()
        print("\nCOREF RESO ENTITY LINKING COMPLETE")
        print(f"Domains Processed:{total_domains}")
        print(f"Documents:{self.statistics.documents_processed}")
        print(f"Clusters:{self.statistics.clusters_created}")
        print(f"Mentions:{self.statistics.mentions_resolved}")
        print(f"Entities Matched:{self.statistics.entities_matched}")
        print(f"Average Confidence:{self.statistics.average_confidence:.4f}")
        self.statistics.print_domain_statistics()


def run_coref_reso_entity_linking():
    CorefResoEntityLinkingPipeline().process_dataset()