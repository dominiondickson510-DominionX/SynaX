# services/api/app/synax_knowledge_graph.py
import json
import os
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

from services.api.app.synax_config import (
    ENTITY_OUTPUT_DIR,
    RELATIONSHIP_OUTPUT_DIR,
    KNOWLEDGE_GRAPH_MANIFEST_DIR,
    BATCH_WRITE_SIZE,
    neo4j_driver,
)
from services.api.app.synax_relationship_extraction import Relationship


@dataclass(slots=True)
class KnowledgeGraphNode:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_entity(cls, entity: Dict[str, Any]):
        source = entity.get("source", "")
        document_id = entity.get("document_id", "")
        provenance = []
        if source or document_id:
            provenance.append({"source": source, "document_id": document_id})
        return cls(
            entity_id=entity["entity_id"],
            canonical_name=entity["canonical_name"],
            entity_type=entity["entity_type"],
            aliases=sorted(set(entity.get("aliases", []))),
            sources=[source] if source else [],
            provenance=provenance,
            metadata=entity.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "sources": self.sources,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }


def load_node_objects(entities: List[Dict[str, Any]]) -> List[KnowledgeGraphNode]:
    return [KnowledgeGraphNode.from_entity(entity) for entity in entities]


def load_relationship_objects(
    relationships: List[Dict[str, Any]],
) -> List[Relationship]:
    return [Relationship.from_dict(rel) for rel in relationships]


class Neo4jConnection:
    def __init__(self):
        self.driver = neo4j_driver

    def execute(self, query: str, parameters: dict | None = None):
        with self.driver.session() as session:
            session.run(query, parameters or {})

    def execute_write(self, callback, *args, **kwargs):
        with self.driver.session() as session:
            return session.execute_write(callback, *args, **kwargs)


class KnowledgeGraphBuilder:
    def __init__(self):
        self.db = Neo4jConnection()
        self._create_indexes()

    def _create_indexes(self):
        queries = [
            """CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE""",
            """CREATE INDEX entity_type_index IF NOT EXISTS FOR (n:Entity) ON (n.entity_type)""",
            """CREATE INDEX canonical_name_index IF NOT EXISTS FOR (n:Entity) ON (n.canonical_name)""",
            """CREATE INDEX sources_index IF NOT EXISTS FOR (n:Entity) ON (n.sources)""",
            """CREATE CONSTRAINT relationship_key_unique IF NOT EXISTS FOR ()-[r:RELATED_TO]-() REQUIRE r.relationship_key IS UNIQUE""",
            """CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.relationship_type)""""",
        ]
        for query in queries:
            self.db.execute(query)
        print("[Knowledge Graph] Neo4j schema ready.")

    @staticmethod
    def _merge_nodes(tx, nodes: List[KnowledgeGraphNode]):
        query = """UNWIND $nodes AS node
        MERGE (n:Entity {entity_id:node.entity_id})
        SET n.canonical_name=node.canonical_name,n.entity_type=node.entity_type,n.metadata=coalesce(n.metadata,{})+coalesce(node.metadata,{})
        SET n.aliases=apoc.coll.toSet(coalesce(n.aliases,[])+coalesce(node.aliases,[]))
        SET n.sources=apoc.coll.toSet(coalesce(n.sources,[])+coalesce(node.sources,[]))
        SET n.provenance=apoc.coll.toSet(coalesce(n.provenance,[])+coalesce(node.provenance,[]))"""
        tx.run(query, nodes=[node.to_dict() for node in nodes])

    @staticmethod
    def _merge_relationships(tx, relationships: List[Relationship]):
        query = """UNWIND $relationships AS rel
MATCH (a:Entity {entity_id:rel.source_entity})
MATCH (b:Entity {entity_id:rel.target_entity})
MERGE (a)-[r:RELATED_TO {relationship_key:rel.relationship_key}]->(b)
SET r.relationship_type=rel.relationship_type,r.evidences=apoc.coll.toSet(coalesce(r.evidences,[])+coalesce(rel.evidences,[])),r.metadata=coalesce(r.metadata,{})+coalesce(rel.metadata,{})"""
        tx.run(
            query, relationships=[relationship.to_dict() for relationship in relationships]
        )

    def insert_nodes(self, nodes: List[KnowledgeGraphNode]):
        for i in range(0, len(nodes), BATCH_WRITE_SIZE):
            self.db.execute_write(self._merge_nodes, nodes[i : i + BATCH_WRITE_SIZE])

    def insert_relationships(self, relationships: List[Relationship]):
        for i in range(0, len(relationships), BATCH_WRITE_SIZE):
            self.db.execute_write(
                self._merge_relationships, relationships[i : i + BATCH_WRITE_SIZE]
            )


class KnowledgeGraphPipeline:
    def __init__(self):
        self.builder = KnowledgeGraphBuilder()

    def load_entities(self, filepath: str) -> List[KnowledgeGraphNode]:
        with open(filepath, "r", encoding="utf-8") as f:
            entities = json.load(f)
        return load_node_objects(entities)

    def load_relationships(self, filepath: str) -> List[Relationship]:
        with open(filepath, "r", encoding="utf-8") as f:
            return load_relationship_objects(json.load(f))

    def compute_graph_hash(
        self, nodes: List[KnowledgeGraphNode], relationships: List[Relationship]
    ) -> str:
        payload = {
            "nodes": [node.to_dict() for node in nodes],
            "relationships": [relationship.to_dict() for relationship in relationships],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def manifest_path(self, domain: str, filename: str) -> str:
        directory = os.path.join(KNOWLEDGE_GRAPH_MANIFEST_DIR, domain)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename + ".graphhash.json")

    def needs_processing(self, domain: str, filename: str, graph_hash: str) -> bool:
        manifest = self.manifest_path(domain, filename)
        if not os.path.exists(manifest):
            return True
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("graph_hash") != graph_hash
        except Exception:
            return True

    def save_manifest(self, domain: str, filename: str, graph_hash: str):
        manifest = self.manifest_path(domain, filename)
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump({"graph_hash": graph_hash}, f, ensure_ascii=False, indent=2)

    def process_document(
        self,
        entity_path: str,
        relationship_path: str,
        domain: str,
        filename: str,
    ):
        nodes = self.load_entities(entity_path)
        relationships = self.load_relationships(relationship_path)
        graph_hash = self.compute_graph_hash(nodes, relationships)
        if not self.needs_processing(
            domain=domain, filename=filename, graph_hash=graph_hash
        ):
            print(f"[Skipped] {domain}/{filename}")
            return
        self.builder.insert_nodes(nodes)
        self.builder.insert_relationships(relationships)
        self.save_manifest(domain=domain, filename=filename, graph_hash=graph_hash)
        print(
            f"[Knowledge Graph] {domain}/{filename} → {len(nodes)} nodes | {len(relationships)} relationships"
        )

    def process_dataset(self):
        for domain in os.listdir(ENTITY_OUTPUT_DIR):
            entity_dir = os.path.join(ENTITY_OUTPUT_DIR, domain)
            relationship_dir = os.path.join(RELATIONSHIP_OUTPUT_DIR, domain)
            if not os.path.isdir(entity_dir):
                continue
            if not os.path.isdir(relationship_dir):
                continue
            print(f"\n{domain.upper()} KNOWLEDGE GRAPH")
            for file in os.listdir(entity_dir):
                if not file.endswith(".entities.json"):
                    continue
                filename = file.removesuffix(".entities.json")
                entity_path = os.path.join(entity_dir, file)
                relationship_path = os.path.join(
                    relationship_dir, filename + ".relationships.json"
                )
                if not os.path.exists(relationship_path):
                    continue
                try:
                    self.process_document(
                        entity_path=entity_path,
                        relationship_path=relationship_path,
                        domain=domain,
                        filename=filename,
                    )
                except Exception as e:
                    print(
                        f"[Knowledge Graph Error] {domain}/{filename}: {e}"
                    )
        print("\nKNOWLEDGE GRAPH COMPLETE")


def run_knowledge_graph():
    KnowledgeGraphPipeline().process_dataset()