# services/api/app/synax_entity_extraction.py
import os
import json
import hashlib
import re
import unicodedata
import spacy
import scispacy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from services.api.app.synax_config import DATA_DIR, ENTITY_OUTPUT_DIR

try:
    spacy_nlp = spacy.load("en_core_web_trf")
    scispacy_nlp = spacy.load("en_ner_bionlp13cg_md")
except Exception as e:
    raise RuntimeError(f"Failed to load spaCy models: {e}")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def make_entity_id(name: str, entity_type: str) -> str:
    key = f"{entity_type}:{normalize_text(name).casefold()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


@dataclass(slots=True)
class Entity:
    entity_id: str
    text: str
    canonical_name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = ""
    document_id: str = ""
    extractor: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "text": self.text,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "confidence": self.confidence,
            "source": self.source,
            "document_id": self.document_id,
            "extractor": self.extractor,
            "metadata": self.metadata,
        }


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    ENTITY = "ENTITY"
    LOCATION = "LOCATION"
    FACILITY = "FACILITY"
    IDENTITY = "IDENTITY"
    LANGUAGE = "LANGUAGE"
    CONCEPT = "CONCEPT"
    TOPIC = "TOPIC"
    EVENT = "EVENT"
    MEDICAL = "MEDICAL"
    LAW = "LAW"
    PRODUCT = "PRODUCT"
    WORK = "WORK"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    QUANTITY = "QUANTITY"
    GENE = "GENE"
    PROTEIN = "PROTEIN"
    CHEMICAL = "CHEMICAL"
    DISEASE = "DISEASE"
    CELL = "CELL"
    CELL_COMPONENT = "CELL_COMPONENT"
    ANATOMY = "ANATOMY"
    ORGANISM = "ORGANISM"
    AMINO_ACID = "AMINO_ACID"


SPACY_ENTITY_MAP: Dict[str, EntityType] = {
    "PERSON": EntityType.PERSON,
    "ORG": EntityType.ORGANIZATION,
    "GPE": EntityType.LOCATION,
    "LOC": EntityType.LOCATION,
    "FAC": EntityType.FACILITY,
    "NORP": EntityType.IDENTITY,
    "EVENT": EntityType.EVENT,
    "WORK_OF_ART": EntityType.WORK,
    "PRODUCT": EntityType.PRODUCT,
    "LAW": EntityType.LAW,
    "LANGUAGE": EntityType.LANGUAGE,
    "DATE": EntityType.DATE,
    "TIME": EntityType.TIME,
    "MONEY": EntityType.MONEY,
    "PERCENT": EntityType.PERCENT,
    "QUANTITY": EntityType.QUANTITY,
}

SCISPACY_ENTITY_MAP: Dict[str, EntityType] = {
    "AMINO_ACID": EntityType.AMINO_ACID,
    "ANATOMICAL_SYSTEM": EntityType.ANATOMY,
    "CANCER": EntityType.DISEASE,
    "CELL": EntityType.CELL,
    "CELLULAR_COMPONENT": EntityType.CELL_COMPONENT,
    "DEVELOPING_ANATOMICAL_STRUCTURE": EntityType.ANATOMY,
    "GENE_OR_GENE_PRODUCT": EntityType.GENE,
    "IMMATERIAL_ANATOMICAL_ENTITY": EntityType.ANATOMY,
    "MULTI_TISSUE_STRUCTURE": EntityType.ANATOMY,
    "ORGAN": EntityType.ANATOMY,
    "ORGANISM": EntityType.ORGANISM,
    "ORGANISM_SUBDIVISION": EntityType.ORGANISM,
    "ORGANISM_SUBSTANCE": EntityType.ORGANISM,
    "PATHOLOGICAL_FORMATION": EntityType.DISEASE,
    "PROTEIN": EntityType.PROTEIN,
    "SIMPLE_CHEMICAL": EntityType.CHEMICAL,
    "TISSUE": EntityType.ANATOMY,
}


def canonical_entity_type(entity_type: Any) -> str:
    if isinstance(entity_type, EntityType):
        return entity_type.value
    label = str(entity_type).strip().upper().replace("-", "_")
    if label in SPACY_ENTITY_MAP:
        return SPACY_ENTITY_MAP[label].value
    if label in SCISPACY_ENTITY_MAP:
        return SCISPACY_ENTITY_MAP[label].value
    return EntityType.ENTITY.value


class BaseExtractor(ABC):
    name: str = "base"

    @abstractmethod
    def batch_extract(
        self, inputs: List[Dict[str, Any]]
    ) -> List[List[Entity]]:
        pass


class SpacyExtractor(BaseExtractor):
    name: str = "spacy"

    def batch_extract(
        self, inputs: List[Dict[str, Any]]
    ) -> List[List[Entity]]:
        texts = [inp["text"] for inp in inputs]
        contexts = [
            {
                "idx": i,
                "domain": inp.get("domain", ""),
                "source": inp.get("source", ""),
                "document_id": inp.get("document_id", ""),
                "metadata": inp.get("metadata", {}),
            }
            for i, inp in enumerate(inputs)
        ]
        results: List[List[Entity]] = [[] for _ in range(len(inputs))]

        for doc, ctx in spacy_nlp.pipe(
            zip(texts, contexts), as_tuples=True, batch_size=64
        ):
            entities: List[Entity] = []
            for ent in doc.ents:
                canonical = normalize_text(ent.text)
                ent_type = canonical_entity_type(ent.label_)
                entities.append(
                    Entity(
                        entity_id=make_entity_id(canonical, ent_type),
                        text=ent.text,
                        canonical_name=canonical,
                        entity_type=ent_type,
                        confidence=0.80,
                        source=ctx["source"],
                        document_id=ctx["document_id"],
                        extractor=self.name,
                        metadata={
                            "domain": ctx["domain"],
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "spacy_label": ent.label_,
                        },
                    )
                )
            results[ctx["idx"]] = entities
        return results


class SciSpacyExtractor(BaseExtractor):
    name: str = "scispacy"

    def batch_extract(
        self, inputs: List[Dict[str, Any]]
    ) -> List[List[Entity]]:
        texts = [inp["text"] for inp in inputs]
        contexts = [
            {
                "idx": i,
                "domain": inp.get("domain", ""),
                "source": inp.get("source", ""),
                "document_id": inp.get("document_id", ""),
                "metadata": inp.get("metadata", {}),
            }
            for i, inp in enumerate(inputs)
        ]
        results: List[List[Entity]] = [[] for _ in range(len(inputs))]

        for doc, ctx in scispacy_nlp.pipe(
            zip(texts, contexts), as_tuples=True, batch_size=64
        ):
            entities: List[Entity] = []
            for ent in doc.ents:
                canonical = normalize_text(ent.text)
                ent_type = canonical_entity_type(ent.label_)
                entities.append(
                    Entity(
                        entity_id=make_entity_id(canonical, ent_type),
                        text=ent.text,
                        canonical_name=canonical,
                        entity_type=ent_type,
                        confidence=0.90,
                        source=ctx["source"],
                        document_id=ctx["document_id"],
                        extractor=self.name,
                        metadata={
                            "domain": ctx["domain"],
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "scispacy_label": ent.label_,
                        },
                    )
                )
            results[ctx["idx"]] = entities
        return results


class MetadataExtractor(BaseExtractor):
    name: str = "metadata"

    def _add_entity(
        self,
        entities: List[Entity],
        text: str,
        entity_type: EntityType,
        source: str,
        document_id: str,
        confidence: float,
        metadata: Dict[str, Any],
        aliases: List[str] | None = None,
        entity_id: str | None = None,
    ):
        text = normalize_text(text)
        if not text:
            return
        entities.append(
            Entity(
                entity_id=entity_id or make_entity_id(text, entity_type.value),
                text=text,
                canonical_name=text,
                entity_type=entity_type.value,
                aliases=aliases or [],
                confidence=confidence,
                source=source,
                document_id=document_id,
                extractor=self.name,
                metadata=metadata,
            )
        )

    def batch_extract(
        self, inputs: List[Dict[str, Any]]
    ) -> List[List[Entity]]:
        results = []
        for inp in inputs:
            domain = inp.get("domain", "")
            source = inp.get("source", "")
            document_id = inp.get("document_id", "")
            meta = inp.get("metadata") or {}
            entities = []
            src = source

            if src == "openalex":
                common = {
                    "title": meta.get("title", ""),
                    "doi": meta.get("doi", ""),
                    "publication_year": meta.get("publication_year"),
                }
                for author in meta.get("authors", []):
                    self._add_entity(
                        entities,
                        author,
                        EntityType.PERSON,
                        source,
                        document_id,
                        1.0,
                        common,
                    )
                for institution in meta.get("institutions", []):
                    self._add_entity(
                        entities,
                        institution,
                        EntityType.ORGANIZATION,
                        source,
                        document_id,
                        1.0,
                        common,
                    )
                for concept in meta.get("concepts", []):
                    self._add_entity(
                        entities,
                        concept,
                        EntityType.CONCEPT,
                        source,
                        document_id,
                        0.95,
                        common,
                    )

            elif src == "pubmed":
                common = {
                    "pmid": meta.get("pmid", ""),
                    "pmcid": meta.get("pmcid", ""),
                    "journal": meta.get("journal", ""),
                    "publication_date": meta.get("publication_date", ""),
                    "doi": meta.get("doi", ""),
                }
                for author in meta.get("authors", []):
                    self._add_entity(
                        entities,
                        author,
                        EntityType.PERSON,
                        source,
                        document_id,
                        1.0,
                        common,
                    )
                for mesh in meta.get("mesh_terms", []):
                    self._add_entity(
                        entities,
                        mesh,
                        EntityType.MEDICAL,
                        source,
                        document_id,
                        1.0,
                        common,
                    )

            elif src == "pubmedcentral":
                common = {
                    "pmid": meta.get("pmid", ""),
                    "pmcid": meta.get("pmcid", ""),
                }
                for author in meta.get("authors", []):
                    self._add_entity(
                        entities,
                        author,
                        EntityType.PERSON,
                        source,
                        document_id,
                        1.0,
                        common,
                    )

            elif src == "arxiv":
                common = {
                    "entry_id": meta.get("entry_id"),
                    "doi": meta.get("doi", ""),
                    "published": meta.get("published", ""),
                    "updated": meta.get("updated", ""),
                }
                for author in meta.get("authors", []):
                    self._add_entity(
                        entities,
                        author,
                        EntityType.PERSON,
                        source,
                        document_id,
                        1.0,
                        common,
                    )
                for category in meta.get("categories", []):
                    self._add_entity(
                        entities,
                        category,
                        EntityType.TOPIC,
                        source,
                        document_id,
                        0.95,
                        common,
                    )

            elif src == "clinicaltrials":
                common = {"nct_id": meta.get("nct_id", "")}
                sponsor = meta.get("sponsor", "")
                if sponsor:
                    self._add_entity(
                        entities,
                        sponsor,
                        EntityType.ORGANIZATION,
                        source,
                        document_id,
                        1.0,
                        common,
                    )
                for condition in meta.get("conditions", []):
                    self._add_entity(
                        entities,
                        condition,
                        EntityType.DISEASE,
                        source,
                        document_id,
                        1.0,
                        common,
                    )
                for keyword in meta.get("keywords", []):
                    self._add_entity(
                        entities,
                        keyword,
                        EntityType.CONCEPT,
                        source,
                        document_id,
                        0.95,
                        common,
                    )

            elif src == "wikidata":
                labels = meta.get("labels", {})
                aliases = meta.get("aliases", {})
                descriptions = meta.get("descriptions", {})
                canonical = normalize_text(labels.get("en"))
                if canonical:
                    alias_list = [
                        normalize_text(alias)
                        for alias in aliases.get("en", [])
                        if normalize_text(alias)
                    ]
                    self._add_entity(
                        entities,
                        canonical,
                        EntityType.ENTITY,
                        source,
                        document_id,
                        1.0,
                        {
                            "qid": meta.get("qid", ""),
                            "labels": labels,
                            "descriptions": descriptions,
                        },
                        aliases=alias_list,
                        entity_id=meta.get("qid")
                        or make_entity_id(canonical, "ENTITY"),
                    )

            results.append(entities)
        return results


class EntityExtractor:
    def __init__(self) -> None:
        self.extractors: List[BaseExtractor] = [
            MetadataExtractor(),
            SpacyExtractor(),
            SciSpacyExtractor(),
        ]

    def merge_entities(self, entities: List[Entity]) -> List[Entity]:
        merged: Dict[str, Entity] = {}
        specialized_types = {
            EntityType.GENE.value,
            EntityType.PROTEIN.value,
            EntityType.CHEMICAL.value,
            EntityType.DISEASE.value,
            EntityType.CELL.value,
            EntityType.CELL_COMPONENT.value,
            EntityType.ANATOMY.value,
            EntityType.ORGANISM.value,
            EntityType.AMINO_ACID.value,
        }

        def type_priority(entity_type: str) -> int:
            if entity_type in specialized_types:
                return 3
            if entity_type == EntityType.ENTITY.value:
                return 1
            return 2

        for entity in entities:
            entity.canonical_name = normalize_text(entity.canonical_name)
            entity.text = normalize_text(entity.text)
            entity.aliases = [
                normalize_text(alias)
                for alias in entity.aliases
                if normalize_text(alias)
            ]

            if not entity.canonical_name:
                continue

            key = entity.canonical_name.casefold()

            if key not in merged:
                entity.metadata.setdefault("extractors", [])
                if entity.extractor not in entity.metadata["extractors"]:
                    entity.metadata["extractors"].append(entity.extractor)
                merged[key] = entity
                continue

            existing = merged[key]

            if entity.text and entity.text != existing.canonical_name:
                if entity.text not in existing.aliases:
                    existing.aliases.append(entity.text)

            for alias in entity.aliases:
                if alias and alias not in existing.aliases:
                    existing.aliases.append(alias)

            extractors_list: List[str] = existing.metadata.setdefault(
                "extractors", []
            )
            if entity.extractor not in extractors_list:
                extractors_list.append(entity.extractor)

            existing.confidence = max(existing.confidence, entity.confidence)

            if type_priority(entity.entity_type) > type_priority(
                existing.entity_type
            ):
                existing.entity_type = entity.entity_type

            for k, v in entity.metadata.items():
                if k == "extractors":
                    continue
                if k not in existing.metadata:
                    existing.metadata[k] = v

        return list(merged.values())

    def score_entity(self, entity: Entity) -> Entity:
        score = 0.0
        extractors: List[str] = entity.metadata.get("extractors", [])

        if "metadata" in extractors:
            score += 0.45
        if "spacy" in extractors:
            score += 0.35
        if "scispacy" in extractors:
            score += 0.35

        score = min(score, 1.0)
        entity.confidence = max(entity.confidence, score)
        return entity

    def sort_entities(self, entities: List[Entity]) -> List[Entity]:
        return sorted(
            entities,
            key=lambda x: (x.confidence, len(x.canonical_name)),
            reverse=True,
        )

    def batch_extract(
        self, inputs: List[Dict[str, Any]]
    ) -> List[List[Entity]]:
        grouped_entities: List[List[Entity]] = [
            [] for _ in range(len(inputs))
        ]

        for extractor in self.extractors:
            try:
                batch_results = extractor.batch_extract(inputs)
                for i, res in enumerate(batch_results):
                    grouped_entities[i].extend(res)
            except Exception as e:
                print(f"[{extractor.name}] {e}")

        final_results: List[List[Entity]] = []
        for entity_pool in grouped_entities:
            merged = self.merge_entities(entity_pool)
            scored = [self.score_entity(entity) for entity in merged]
            final_results.append(self.sort_entities(scored))

        return final_results


class EntityExtractionPipeline:
    def __init__(self):
        self.extractor = EntityExtractor()
        self.batch_size = 64

    def load_document(self, filepath: str) -> tuple[str, Dict[str, Any]]:
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
                                            str(v)
                                            for v in row.values()
                                            if v
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

    def compute_document_hash(self, text: str) -> str:
        return hashlib.sha256(
            normalize_text(text).encode("utf-8")
        ).hexdigest()

    def build_input(self, filepath: str, domain: str) -> Dict[str, Any]:
        text, metadata = self.load_document(filepath)
        filename = os.path.splitext(os.path.basename(filepath))[0]
        relative_path = os.path.relpath(filepath, DATA_DIR)
        document_hash = self.compute_document_hash(text)
        source = metadata.get("source")
        document_id = hashlib.sha256(
            f"{source}:{domain}:{relative_path}".encode("utf-8")
        ).hexdigest()[:32]

        return {
            "text": text,
            "domain": domain,
            "source": source,
            "document_id": document_id,
            "metadata": metadata,
            "document_hash": document_hash,
        }

    def needs_processing(
        self, domain: str, filename: str, document_hash: str
    ) -> bool:
        entity_path = os.path.join(
            ENTITY_OUTPUT_DIR, domain, filename + ".entities.json"
        )
        if not os.path.exists(entity_path):
            return True

        try:
            with open(entity_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return True
            if not data.get("entities"):
                return True
            return data.get("document_hash") != document_hash
        except Exception:
            return True

    def save_entities(
        self, domain: str, filename: str, entities, document
    ):
        domain_dir = os.path.join(ENTITY_OUTPUT_DIR, domain)
        os.makedirs(domain_dir, exist_ok=True)
        output_path = os.path.join(
            domain_dir, filename + ".entities.json"
        )

        payload = {
            "document_id": document["document_id"],
            "document_hash": document["document_hash"],
            "metadata": document["metadata"],
            "entities": [entity.to_dict() for entity in entities],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def process_batch(self, documents, filenames, domain):
        try:
            entity_results = self.extractor.batch_extract(documents)
            total_entities = 0

            for document, filename, entities in zip(
                documents, filenames, entity_results
            ):
                document_hash = document["document_hash"]
                for entity in entities:
                    entity.metadata["document_hash"] = document_hash

                self.save_entities(
                    domain=domain,
                    filename=filename,
                    entities=entities,
                    document=document,
                )
                total_entities += len(entities)
                print(
                    f"[EntityExtraction] {domain}/{filename}\n{len(entities)} entities"
                )

            return total_entities
        except Exception as e:
            print(f"[Batch Extraction Error] {e}")
            return 0

    def process_directory(self, directory: str, domain: str):
        if not os.path.exists(directory):
            print(f"[Skipped] {directory} does not exist.")
            return 0, 0

        total_documents = 0
        total_entities = 0
        batch_documents = []
        batch_filenames = []

        for root, _, files in os.walk(directory):
            for file in files:
                if not file.endswith((".txt", ".json")):
                    continue

                filepath = os.path.join(root, file)

                try:
                    document = self.build_input(
                        filepath=filepath, domain=domain
                    )
                    filename = os.path.splitext(file)[0]

                    if not self.needs_processing(
                        domain=domain,
                        filename=filename,
                        document_hash=document["document_hash"],
                    ):
                        print(f"[Skipped] {domain}/{filename}")
                        continue

                    batch_documents.append(document)
                    batch_filenames.append(filename)
                    total_documents += 1

                    if len(batch_documents) == self.batch_size:
                        total_entities += self.process_batch(
                            documents=batch_documents,
                            filenames=batch_filenames,
                            domain=domain,
                        )
                        batch_documents = []
                        batch_filenames = []

                except Exception as e:
                    print(f"[Entity Extraction Error] {filepath}: {e}")

        if batch_documents:
            total_entities += self.process_batch(
                documents=batch_documents,
                filenames=batch_filenames,
                domain=domain,
            )

        print(
            f"[{domain}] Processed {total_documents} documents | Extracted {total_entities} entities"
        )
        return total_documents, total_entities

    def process_dataset(self):
        total_domains = 0
        total_documents = 0
        total_entities = 0

        for domain in os.listdir(DATA_DIR):
            domain_path = os.path.join(DATA_DIR, domain)

            if (
                not os.path.isdir(domain_path)
                or domain in {"entities"}
            ):
                continue

            print(f"\n{domain.upper()}")
            processed_documents, extracted_entities = self.process_directory(
                directory=domain_path, domain=domain
            )
            total_domains += 1
            total_documents += processed_documents
            total_entities += extracted_entities

        print(
            f"\nEntity Extraction Complete\nDomains Processed: {total_domains}\nDocuments Processed: {total_documents}\nEntities Extracted: {total_entities}"
        )
        
def run_entity_extraction():
    return EntityExtractionPipeline().process_dataset()