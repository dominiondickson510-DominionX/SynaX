# services/api/app/synax_knowledge_graph_retrieval.py
from enum import Enum
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from services.api.app.synax_config import neo4j_driver


class Neo4jConnection:
    def __init__(self):
        self.driver = neo4j_driver

    def execute_read(self, callback, *args, **kwargs):
        with self.driver.session() as session:
            return session.execute_read(callback, *args, **kwargs)


@dataclass(slots=True)
class EntityRequest:
    canonical_name: str
    entity_type: Optional[str] = None


class RelationshipDirection(str, Enum):
    INCOMING = "incoming"
    OUTGOING = "outgoing"
    BOTH = "both"


@dataclass(slots=True)
class RelationshipConstraint:
    relationship_type: Optional[str] = None
    direction: RelationshipDirection = RelationshipDirection.BOTH


@dataclass(slots=True)
class NeighborhoodRequest:
    hops: int = 4
    max_nodes: int = 150
    max_relationships: int = 350
    direction: RelationshipDirection = RelationshipDirection.BOTH


@dataclass(slots=True)
class RetrievalRequest:
    entities: List[EntityRequest]
    relationship: RelationshipConstraint = field(
        default_factory=RelationshipConstraint
    )
    neighborhood: NeighborhoodRequest = field(default_factory=NeighborhoodRequest)


@dataclass(slots=True)
class ResolvedEntity:
    entity_id: str
    canonical_name: str
    entity_type: str


class EntityResolver:
    def __init__(self):
        self.db = Neo4jConnection()

    @staticmethod
    def _resolve_entity(tx, request: EntityRequest):
        query = """MATCH (n:Entity)
WHERE (
    toLower(n.canonical_name) = toLower($canonical_name)
    OR ANY(alias IN coalesce(n.aliases, []) WHERE toLower(alias) = toLower($canonical_name))
)
AND ($entity_type IS NULL OR n.entity_type = $entity_type)
RETURN n"""
        result = tx.run(
            query,
            canonical_name=request.canonical_name,
            entity_type=request.entity_type,
        )
        return [record["n"] for record in result]

    def resolve(self, requests: List[EntityRequest]) -> List[ResolvedEntity]:
        resolved = []
        seen = set()
        for request in requests:
            nodes = self.db.execute_read(self._resolve_entity, request)
            for node in nodes:
                entity_id = node["entity_id"]
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                resolved.append(
                    ResolvedEntity(
                        entity_id=node["entity_id"],
                        canonical_name=node["canonical_name"],
                        entity_type=node["entity_type"],
                    )
                )
        return resolved


@dataclass(slots=True)
class RetrievedNode:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    raw_node: Dict[str, Any] = field(default_factory=dict)


class NodeRetriever:
    def __init__(self):
        self.db = Neo4jConnection()

    @staticmethod
    def _retrieve_nodes_by_ids(tx, entity_ids: List[str]):
        query = """MATCH (n:Entity)
WHERE n.entity_id IN $entity_ids
RETURN properties(n) AS node"""
        return list(tx.run(query, entity_ids=entity_ids))

    def load_by_ids(self, entity_ids: List[str]) -> List[RetrievedNode]:
        if not entity_ids:
            return []
        records = self.db.execute_read(self._retrieve_nodes_by_ids, entity_ids)
        return [
            RetrievedNode(
                entity_id=node["entity_id"],
                canonical_name=node["canonical_name"],
                entity_type=node["entity_type"],
                aliases=list(node.get("aliases", [])),
                metadata=dict(node.get("metadata", {})),
                sources=list(node.get("sources", [])),
                provenance=list(node.get("provenance", [])),
                raw_node=node,
            )
            for node in (record["node"] for record in records)
        ]

    def retrieve(self, resolved_entities: List[ResolvedEntity]) -> List[RetrievedNode]:
        if not resolved_entities:
            return []
        return self.load_by_ids([entity.entity_id for entity in resolved_entities])


@dataclass(slots=True)
class RetrievedRelationship:
    relationship_key: str
    relationship_type: str
    source_entity: str
    target_entity: str
    evidences: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_relationship: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedNeighborhood:
    nodes: List[RetrievedNode]
    relationships: List[RetrievedRelationship]


class NeighborhoodRetriever:
    def __init__(self):
        self.db = Neo4jConnection()

    @staticmethod
    def _expand_frontier(
        tx,
        frontier_ids: List[str],
        visited_ids: List[str],
        constraint: RelationshipConstraint,
    ):
        if constraint.direction is RelationshipDirection.BOTH:
            pattern = "(current)-[r]-(neighbor)"
        elif constraint.direction is RelationshipDirection.OUTGOING:
            pattern = "(current)-[r]->(neighbor)"
        elif constraint.direction is RelationshipDirection.INCOMING:
            pattern = "(current)<-[r]-(neighbor)"
        else:
            return []
        query = f"""MATCH {pattern}
WHERE current.entity_id IN $frontier_ids
AND NOT neighbor.entity_id IN $visited_ids
AND ($relationship_type IS NULL OR r.relationship_type = $relationship_type)
RETURN properties(neighbor) AS node, {{relationship: properties(r), source_entity: startNode(r).entity_id, target_entity: endNode(r).entity_id}} AS relationship"""
        return list(
            tx.run(
                query,
                frontier_ids=frontier_ids,
                visited_ids=visited_ids,
                relationship_type=constraint.relationship_type,
            )
        )

    def retrieve(
        self,
        seed_nodes: List[RetrievedNode],
        constraint: RelationshipConstraint,
        neighborhood: NeighborhoodRequest,
    ) -> RetrievedNeighborhood:
        if not seed_nodes:
            return RetrievedNeighborhood(nodes=[], relationships=[])
        visited_nodes = {}
        visited_relationships = {}
        frontier = []
        for node in seed_nodes:
            visited_nodes[node.entity_id] = node
            frontier.append(node.entity_id)
        for _ in range(neighborhood.hops):
            if not frontier or len(visited_nodes) >= neighborhood.max_nodes:
                break
            records = self.db.execute_read(
                self._expand_frontier,
                frontier,
                list(visited_nodes.keys()),
                constraint,
            )
            next_frontier = []
            for record in records:
                node = record["node"]
                entity_id = node["entity_id"]
                if (
                    entity_id not in visited_nodes
                    and len(visited_nodes) < neighborhood.max_nodes
                ):
                    visited_nodes[entity_id] = RetrievedNode(
                        entity_id=node["entity_id"],
                        canonical_name=node["canonical_name"],
                        entity_type=node["entity_type"],
                        aliases=list(node.get("aliases", [])),
                        metadata=dict(node.get("metadata", {})),
                        sources=list(node.get("sources", [])),
                        provenance=list(node.get("provenance", [])),
                        raw_node=dict(node),
                    )
                    next_frontier.append(entity_id)
                relationship = record["relationship"]
                rel = relationship["relationship"]
                relationship_key = rel["relationship_key"]
                if (
                    relationship_key not in visited_relationships
                    and len(visited_relationships) < neighborhood.max_relationships
                ):
                    visited_relationships[relationship_key] = RetrievedRelationship(
                        relationship_key=relationship_key,
                        relationship_type=rel["relationship_type"],
                        source_entity=relationship["source_entity"],
                        target_entity=relationship["target_entity"],
                        evidences=list(rel.get("evidences", [])),
                        metadata=dict(rel.get("metadata", {})),
                        raw_relationship=dict(rel),
                    )
            frontier = next_frontier
        return RetrievedNeighborhood(
            nodes=list(visited_nodes.values()),
            relationships=list(visited_relationships.values()),
        )


@dataclass(slots=True)
class KnowledgeGraphContext:
    nodes: List[RetrievedNode]
    relationships: List[RetrievedRelationship]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "entity_id": node.entity_id,
                    "canonical_name": node.canonical_name,
                    "entity_type": node.entity_type,
                    "aliases": node.aliases,
                    "metadata": node.metadata,
                    "sources": node.sources,
                    "provenance": node.provenance,
                    "raw_node": node.raw_node,
                }
                for node in self.nodes
            ],
            "relationships": [
                {
                    "relationship_key": relationship.relationship_key,
                    "relationship_type": relationship.relationship_type,
                    "source_entity": relationship.source_entity,
                    "target_entity": relationship.target_entity,
                    "evidences": relationship.evidences,
                    "metadata": relationship.metadata,
                    "raw_relationship": relationship.raw_relationship,
                }
                for relationship in self.relationships
            ],
        }


class ContextBuilder:
    def build(
        self, neighborhood: RetrievedNeighborhood
    ) -> KnowledgeGraphContext:
        return KnowledgeGraphContext(
            nodes=neighborhood.nodes,
            relationships=neighborhood.relationships,
        )


class KnowledgeGraphRetrieval:
    def __init__(self):
        self.entity_resolver = EntityResolver()
        self.node_retriever = NodeRetriever()
        self.neighborhood_retriever = NeighborhoodRetriever()
        self.context_builder = ContextBuilder()

    def retrieve(self, request: RetrievalRequest) -> KnowledgeGraphContext:
        resolved_nodes = self.entity_resolver.resolve(request.entities)
        retrieved_nodes = self.node_retriever.retrieve(resolved_nodes)
        neighborhood = self.neighborhood_retriever.retrieve(
            seed_nodes=retrieved_nodes,
            constraint=request.relationship,
            neighborhood=request.neighborhood,
        )
        return self.context_builder.build(neighborhood=neighborhood)