# services/api/app/synax_relationship_extraction.py
import os
import json
import hashlib
import re
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from services.api.app.synax_entity_extraction import normalize_text
from services.api.app.synax_config import (
    RELATIONSHIP_OUTPUT_DIR,
    COREF_LINKED_ENTITY_OUTPUT_DIR,
)


@dataclass(slots=True)
class RelationshipEvidence:
    document_id: str
    sentence: str
    source: str
    extractor: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Relationship:
    relationship_key: str
    source_entity: str
    relationship_type: str
    target_entity: str
    confidence: float = 0.0
    evidences: List[RelationshipEvidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            relationship_key=data["relationship_key"],
            source_entity=data["source_entity"],
            relationship_type=data["relationship_type"],
            target_entity=data["target_entity"],
            confidence=data.get("confidence", 0.0),
            evidences=[
                RelationshipEvidence(
                    document_id=e["document_id"],
                    sentence=e["sentence"],
                    source=e["source"],
                    extractor=e["extractor"],
                    metadata=e.get("metadata", {}),
                )
                for e in data.get("evidences", [])
            ],
            metadata=data.get("metadata", {}),
        )

    def to_dict(self):
        return {
            "relationship_key": self.relationship_key,
            "source_entity": self.source_entity,
            "relationship_type": self.relationship_type,
            "target_entity": self.target_entity,
            "confidence": self.confidence,
            "evidences": [
                {
                    "document_id": e.document_id,
                    "sentence": e.sentence,
                    "source": e.source,
                    "extractor": e.extractor,
                    "metadata": e.metadata,
                }
                for e in self.evidences
            ],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EntityMention:
    entity_id: str
    canonical_name: str
    entity_type: str
    document_id: str
    source: str = ""
    confidence: float = 0.0
    start: int = 0
    end: int = 0
    text: str = ""


@dataclass(slots=True)
class Sentence:
    text: str
    start: int
    end: int
    mentions: List[EntityMention] = field(default_factory=list)


class SentenceEntityMapper:
    def attach_mentions(
        self, sentences: List[Sentence], mentions: List[Dict[str, Any]]
    ) -> List[Sentence]:
        for sentence in sentences:
            sentence.mentions.clear()

        for mention in mentions:
            index = mention["sentence_index"]
            if index < 0 or index >= len(sentences):
                continue

            sentence = sentences[index]
            sentence.mentions.append(
                EntityMention(
                    entity_id=mention["entity_id"],
                    canonical_name=mention["canonical_name"],
                    entity_type=mention["entity_type"],
                    source=mention.get("source", ""),
                    document_id=mention["document_id"],
                    confidence=mention["confidence"],
                    start=mention["start"] - sentence.start,
                    end=mention["end"] - sentence.start,
                    text=mention["text"],
                )
            )

        for sentence in sentences:
            sentence.mentions.sort(key=lambda m: (m.start, m.end))

        return sentences


class BaseRelationInference(ABC):
    name = "base"

    @abstractmethod
    def extract(self, text: str) -> List[Dict[str, Any]]:
        raise NotImplementedError


@dataclass(slots=True)
class ExtractedTriplet:
    subject: str
    relation: str
    object: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


def _normalize_name(text: str) -> str:
    text = normalize_text(text).casefold()
    text = text.replace("'", "'").replace("'", "'").replace("`", "'")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return text


class RebelInference(BaseRelationInference):
    name = "rebel"

    def __init__(self, model_name: str = "Babelscape/rebel-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def extract(self, text: str) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        generated = self.model.generate(
            **encoded,
            max_length=256,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        decoded = self.tokenizer.batch_decode(
            generated, skip_special_tokens=False
        )[0]

        return [t.to_dict() for t in self.parse_generated_text(decoded)]

    def parse_generated_text(self, generated_text: str) -> List[ExtractedTriplet]:
        generated_text = _normalize_name(generated_text)
        if not generated_text:
            return []

        tokens = generated_text.split()
        triplets: List[ExtractedTriplet] = []
        current_subject: List[str] = []
        current_relation: List[str] = []
        current_object: List[str] = []
        state: Optional[str] = None

        def flush_triplet():
            nonlocal current_subject, current_relation, current_object
            subject = " ".join(current_subject).strip()
            relation = " ".join(current_relation).strip()
            obj = " ".join(current_object).strip()

            if subject and relation and obj:
                triplets.append(
                    ExtractedTriplet(
                        subject=subject,
                        relation=relation.upper().replace(" ", "_"),
                        object=obj,
                        confidence=1.0,
                        metadata={"engine": self.name},
                    )
                )
            current_subject = []
            current_relation = []
            current_object = []

        for token in tokens:
            if token == "<triplet>":
                flush_triplet()
                state = None
                continue

            if token == "<subj>":
                state = "subject"
                continue

            if token == "<obj>":
                state = "object"
                continue

            if token.startswith("<") and token.endswith(">"):
                relation = token[1:-1].strip()
                if relation:
                    current_relation = [relation]
                state = "relation"
                continue

            if state == "subject":
                current_subject.append(token)
            elif state == "relation":
                current_relation.append(token)
            elif state == "object":
                current_object.append(token)

        flush_triplet()
        return triplets


class MentionAligner:
    def _candidate_score(self, generated: str, mention: EntityMention) -> float:
        generated_normalized = _normalize_name(generated)
        mention_text = _normalize_name(mention.text)
        canonical_name = _normalize_name(mention.canonical_name)

        if not generated_normalized:
            return 0.0

        if generated_normalized == canonical_name:
            return 1.00

        if generated_normalized == mention_text:
            return 0.98

        generated_tokens = re.findall(
            r"[a-z0-9]+(?:['-][a-z0-9]+)*", generated_normalized
        )
        if not generated_tokens:
            return 0.0

        generated_token_set = set(generated_tokens)
        for candidate in (canonical_name, mention_text):
            if not candidate:
                continue

            candidate_tokens = re.findall(
                r"[a-z0-9]+(?:['-][a-z0-9]+)*", candidate
            )
            if not candidate_tokens:
                continue

            candidate_token_set = set(candidate_tokens)
            if generated_token_set == candidate_token_set:
                return 0.90

            if (
                len(generated_tokens) > 1
                and generated_token_set.issubset(candidate_token_set)
            ):
                return 0.82

        return 0.0

    def _rank_candidates(
        self, generated: str, mentions: List[EntityMention]
    ) -> List[Tuple[float, EntityMention]]:
        ranked = [
            (self._candidate_score(generated, mention), mention)
            for mention in mentions
        ]
        ranked = [candidate for candidate in ranked if candidate[0] >= 0.70]
        ranked.sort(
            key=lambda candidate: (
                candidate[0],
                candidate[1].confidence,
                candidate[1].end - candidate[1].start,
            ),
            reverse=True,
        )
        return ranked

    def align(
        self, generated_text: str, mentions: List[EntityMention]
    ) -> Optional[EntityMention]:
        if not generated_text.strip() or not mentions:
            return None

        ranked = self._rank_candidates(generated_text, mentions)
        if not ranked:
            return None

        best_score, best_mention = ranked[0]

        exact_canonical = [
            mention for score, mention in ranked if score == 1.00
        ]
        if exact_canonical:
            entity_ids = {mention.entity_id for mention in exact_canonical}
            return exact_canonical[0] if len(entity_ids) == 1 else None

        if len(ranked) == 1:
            return best_mention

        second_score, second_mention = ranked[1]
        if best_mention.entity_id == second_mention.entity_id:
            return best_mention

        if best_score - second_score < 0.08:
            return None

        return best_mention


def make_relationship_key(
    source_entity: str, relationship_type: str, target_entity: str
) -> str:
    relationship_type = relationship_type.upper().strip()
    key = f"{source_entity}|{relationship_type}|{target_entity}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class RelationshipExtractor:
    def __init__(self):
        self.entity_mapper = SentenceEntityMapper()
        self.inference = RebelInference()
        self.mention_aligner = MentionAligner()

    def merge_relationships(
        self, relationships: List[Relationship]
    ) -> List[Relationship]:
        merged = {}

        for relationship in relationships:
            key = relationship.relationship_key
            if key not in merged:
                merged[key] = relationship
                continue

            merged_relationship = merged[key]
            existing_count = len(merged_relationship.evidences)
            new_count = len(relationship.evidences)
            total_count = existing_count + new_count

            if total_count:
                merged_relationship.confidence = (
                    (merged_relationship.confidence * existing_count)
                    + (relationship.confidence * new_count)
                ) / total_count

            merged_relationship.evidences.extend(relationship.evidences)

            for k, v in relationship.metadata.items():
                merged_relationship.metadata.setdefault(k, v)

        return list(merged.values())

    def sort_relationships(
        self, relationships: List[Relationship]
    ) -> List[Relationship]:
        return sorted(
            relationships,
            key=lambda r: (r.confidence, len(r.evidences)),
            reverse=True,
        )

    def extract(
        self,
        text: str,
        sentence_data: List[Dict[str, Any]],
        mentions: List[Dict[str, Any]],
    ) -> List[Relationship]:
        sentences = [
            Sentence(text=s["text"], start=s["start"], end=s["end"])
            for s in sentence_data
        ]
        sentences = self.entity_mapper.attach_mentions(sentences, mentions)
        relationships = []

        for sentence in sentences:
            if len(sentence.mentions) < 2:
                continue

            try:
                raw_triplets = self.inference.extract(sentence.text)
                triplets = [
                    ExtractedTriplet(
                        subject=t["subject"],
                        relation=t["relation"],
                        object=t["object"],
                        confidence=t["confidence"],
                        metadata=t.get("metadata", {}),
                    )
                    for t in raw_triplets
                ]

                for triplet in triplets:
                    subject = self.mention_aligner.align(
                        triplet.subject, sentence.mentions
                    )
                    if subject is None:
                        continue

                    obj = self.mention_aligner.align(
                        triplet.object, sentence.mentions
                    )
                    if obj is None:
                        continue

                    evidence_confidence = (
                        0.2 * triplet.confidence
                        + 0.4 * subject.confidence
                        + 0.4 * obj.confidence
                    )

                    relationships.append(
                        Relationship(
                            relationship_key=make_relationship_key(
                                subject.entity_id,
                                triplet.relation,
                                obj.entity_id,
                            ),
                            source_entity=subject.entity_id,
                            relationship_type=triplet.relation,
                            target_entity=obj.entity_id,
                            confidence=evidence_confidence,
                            evidences=[
                                RelationshipEvidence(
                                    document_id=subject.document_id,
                                    sentence=sentence.text,
                                    source=subject.source,
                                    extractor=self.inference.name,
                                    metadata=triplet.metadata,
                                )
                            ],
                        )
                    )
            except Exception:
                pass

        relationships = self.merge_relationships(relationships)
        return self.sort_relationships(relationships)


class RelationshipExtractionPipeline:
    def __init__(self):
        self.extractor = RelationshipExtractor()

    def needs_processing(
        self, domain: str, filename: str, document_hash: str
    ) -> bool:
        relationship_path = os.path.join(
            RELATIONSHIP_OUTPUT_DIR, domain, filename + ".relationships.json"
        )
        if not os.path.exists(relationship_path):
            return True

        try:
            with open(relationship_path, "r", encoding="utf-8") as f:
                relationships = json.load(f)

            if not relationships:
                return True

            return (
                relationships[0].get("metadata", {}).get("document_hash")
                != document_hash
            )
        except Exception:
            return True

    def load_coreference_document(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_relationships(
        self, domain: str, filename: str, relationships: List[Relationship]
    ):
        domain_dir = os.path.join(RELATIONSHIP_OUTPUT_DIR, domain)
        os.makedirs(domain_dir, exist_ok=True)
        output = os.path.join(domain_dir, filename + ".relationships.json")

        with open(output, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in relationships],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def process_document(self, filename: str, coref_path: str, domain: str):
        coref = self.load_coreference_document(coref_path)
        text = coref["rewritten_text"]
        mentions = coref["mentions"]
        sentences = coref["sentences"]
        document_metadata = coref.get("metadata", {})
        document_hash = document_metadata.get("document_hash", "")

        if not self.needs_processing(
            domain=domain, filename=filename, document_hash=document_hash
        ):
            print(f"[Skipped] {domain}/{filename}")
            return 0

        relationships = self.extractor.extract(
            text=text, sentence_data=sentences, mentions=mentions
        )

        for relationship in relationships:
            relationship.metadata = {**document_metadata, **relationship.metadata}

        self.save_relationships(domain, filename, relationships)
        print(
            f"[Relationship Extraction] {domain}/{filename} → {len(relationships)} relationships"
        )
        return len(relationships)

    def process_dataset(self):
        for domain in os.listdir(COREF_LINKED_ENTITY_OUTPUT_DIR):
            coref_dir = os.path.join(COREF_LINKED_ENTITY_OUTPUT_DIR, domain)
            if not os.path.isdir(coref_dir):
                continue

            print(f"\n{domain.upper()} RELATIONSHIP EXTRACTION")

            for file in os.listdir(coref_dir):
                if not file.endswith(".coreflinked.json"):
                    continue

                filename = file.removesuffix(".coreflinked.json")
                coref_path = os.path.join(coref_dir, file)

                try:
                    self.process_document(
                        filename=filename, coref_path=coref_path, domain=domain
                    )
                except Exception as e:
                    print("[Relationship Extraction Error]", e)


def run_relationship_extraction():
    RelationshipExtractionPipeline().process_dataset()