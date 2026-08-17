# services/api/app/synax_query_planner.py
import asyncio
import json
import numpy as np
from abc import ABC
from abc import abstractmethod
from datetime import date
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from services.api.app.synax_config import gpt_client


@dataclass(slots=True)
class SourcePlan:
    source: str
    top_k: int
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "top_k": self.top_k,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class GraphEntity:
    canonical_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"canonical_name": self.canonical_name}


@dataclass(slots=True)
class GraphNeighborhoodPlan:
    hops: int = 4
    max_nodes: int = 300
    max_relationships: int = 750
    direction: str = "both"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hops": self.hops,
            "max_nodes": self.max_nodes,
            "max_relationships": self.max_relationships,
            "direction": self.direction,
        }


@dataclass(slots=True)
class GraphRetrievalPlan:
    entities: List[GraphEntity] = field(default_factory=list)
    neighborhood: GraphNeighborhoodPlan = field(default_factory=GraphNeighborhoodPlan)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "neighborhood": self.neighborhood.to_dict(),
        }


@dataclass(slots=True)
class QueryIntent:
    research_goal: str
    reasoning_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "research_goal": self.research_goal,
            "reasoning_type": self.reasoning_type,
        }


@dataclass(slots=True)
class DateConstraint:
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


@dataclass(slots=True)
class QueryConstraints:
    sources: List[str] = field(default_factory=list)
    excluded_sources: List[str] = field(default_factory=list)
    date_range: DateConstraint = field(default_factory=DateConstraint)
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sources": self.sources,
            "excluded_sources": self.excluded_sources,
            "date_range": self.date_range.to_dict(),
            "language": self.language,
        }


@dataclass(slots=True)
class QueryPlan:
    query: str
    rewritten_query: str
    embedding: np.ndarray
    intent: QueryIntent
    constraints: QueryConstraints
    source_plans: List[SourcePlan]
    graph_plan: GraphRetrievalPlan

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "intent": self.intent.to_dict(),
            "constraints": self.constraints.to_dict(),
            "source_plans": [source.to_dict() for source in self.source_plans],
            "graph_plan": self.graph_plan.to_dict(),
        }


class QueryPlanner(ABC):
    @abstractmethod
    async def plan(self, query: str) -> QueryPlan:
        raise NotImplementedError


@dataclass(slots=True)
class PlannedSource:
    source: str
    top_k: int
    confidence: float

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PlannedSource":
        return cls(
            source=str(obj["source"]),
            top_k=min(50, max(5, int(obj["top_k"]))),
            confidence=max(0.0, min(1.0, float(obj["confidence"]))),
        )


@dataclass(slots=True)
class PlannedGraphEntity:
    canonical_name: str

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PlannedGraphEntity":
        return cls(canonical_name=str(obj["canonical_name"]))


@dataclass(slots=True)
class PlannedNeighborhood:
    hops: int
    max_nodes: int
    max_relationships: int
    direction: str

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PlannedNeighborhood":
        direction = str(obj["direction"])
        if direction not in {"incoming", "outgoing", "both"}:
            direction = "both"
        return cls(
            hops=min(4, max(1, int(obj["hops"]))),
            max_nodes=min(300, max(1, int(obj["max_nodes"]))),
            max_relationships=min(750, max(1, int(obj["max_relationships"]))),
            direction=direction,
        )


@dataclass(slots=True)
class PlannedDateRange:
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Date must be a string in YYYY-MM-DD format.")
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid date '{value}'. Expected YYYY-MM-DD."
            ) from exc

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PlannedDateRange":
        start_date = cls._parse_date(obj.get("start_date"))
        end_date = cls._parse_date(obj.get("end_date"))
        if start_date is not None and end_date is not None:
            if start_date > end_date:
                raise ValueError("start_date cannot be later than end_date.")
        return cls(start_date=start_date, end_date=end_date)


@dataclass(slots=True)
class PlannerResponse:
    rewritten_query: str
    research_goal: str
    reasoning_type: str
    sources: List[PlannedSource]
    graph_entities: List[PlannedGraphEntity]
    graph_neighborhood: PlannedNeighborhood
    date_range: PlannedDateRange
    excluded_sources: List[str] = field(default_factory=list)
    language: str | None = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PlannerResponse":
        return cls(
            rewritten_query=str(obj["rewritten_query"]),
            research_goal=str(obj["research_goal"]),
            reasoning_type=str(obj["reasoning_type"]),
            sources=[
                PlannedSource.from_dict(item) for item in obj.get("sources", [])
            ],
            excluded_sources=[str(item) for item in obj.get("excluded_sources", [])],
            language=obj.get("language"),
            graph_entities=[
                PlannedGraphEntity.from_dict(item)
                for item in obj.get("graph_entities", [])
            ],
            graph_neighborhood=PlannedNeighborhood.from_dict(
                obj["graph_neighborhood"]
            ),
            date_range=PlannedDateRange.from_dict(obj["date_range"]),
        )


class GPTQueryPlanner(QueryPlanner):
    def __init__(self, embedder):
        self.embedder = embedder

    async def plan(self, query: str) -> QueryPlan:
        planner_response = await self._plan_with_gpt(query)
        rewritten_query = planner_response.rewritten_query.strip()
        embedding = await asyncio.to_thread(
            self.embedder.encode,
            rewritten_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._build_query_plan(
            original_query=query,
            planner_response=planner_response,
            embedding=embedding,
        )

    async def _plan_with_gpt(self, query: str) -> PlannerResponse:
        system_prompt = (
            "You are the Query Planning Engine for SynaX. Your ONLY responsibility is to produce a retrieval plan. "
            "You NEVER answer the user's question. You NEVER explain your reasoning. You NEVER generate research summaries, "
            "analysis, or recommendations. You ONLY rewrite the user's query into an optimized semantic retrieval query, "
            "determine the research goal, determine the reasoning type, select the minimum number of knowledge sources needed "
            "from: wikipedia,arxiv,pubmed,pubmedcentral,clinicaltrials,wikidata,openalex, allocate an appropriate top_k "
            "BETWEEN minimum 5 and maximum 50 for each selected source, and exclude unnecessary sources. Every source name "
            "MUST match one of the listed names EXACTLY, character-for-character. Use the exact spelling, exact lowercase "
            "letters, and no spaces, underscores, hyphens, abbreviations, aliases, variations, or alternative spellings. "
            "Never invent or modify a source name. Identify every entity explicitly mentioned or clearly implied in the "
            "rewritten query. For each entity, return ONLY its standardized canonical_name. Determine an appropriate graph "
            "neighborhood by selecting hops, max_nodes, and max_relationships. Choose values appropriate for the complexity "
            "of the research question. hops MUST be between 1 and 4 inclusive. max_nodes MUST be between 1 and 300 inclusive. "
            "max_relationships MUST be between 1 and 750 inclusive. Determine whether the user's query contains a temporal "
            "constraint. Return date_range.start_date and date_range.end_date using the exact ISO 8601 calendar date format "
            "YYYY-MM-DD. If the query specifies a bounded period, return both dates. If the query specifies only a lower "
            "temporal boundary, return start_date and set end_date to null. If the query specifies only an upper temporal "
            "boundary, return end_date and set start_date to null. If the query contains no temporal constraint, set both "
            "values to null. Examples: \"research published between 2018 and 2022\" → start_date: \"2018-01-01\" → "
            "end_date: \"2022-12-31\", \"research since 2020\" → start_date: \"2020-01-01\" → end_date: null, "
            "\"research before 2015\" → start_date: null → end_date: \"2014-12-31\", \"research published in 2023\" → "
            "start_date: \"2023-01-01\" → end_date: \"2023-12-31\", \"No temporal restriction\" → start_date: null → "
            "end_date: null. DO NOT return years, months, natural-language dates, timestamps, relative expressions, or any "
            "other date representation. Also, determine the appropriate graph relationship traversal direction. Use \"outgoing\" "
            "when the research question primarily asks what the identified entities affect, regulate, cause, encode, produce, "
            "interact with, or otherwise connect outward to. Use \"incoming\" when the research question primarily asks what "
            "affects, regulates, causes, encodes, produces, or otherwise connects into the identified entities. Use \"both\" "
            "when understanding the question requires relationships in both directions or when directional restriction would "
            "omit important relevant context. Return EXACTLY ONE of: \"incoming\", \"outgoing\", or \"both\". Return STRICT JSON "
            "that conforms exactly to the provided schema. No markdown. No prose. No additional fields."
        )
        response = await gpt_client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "query_plan",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "rewritten_query": {"type": "string"},
                            "research_goal": {"type": "string"},
                            "reasoning_type": {"type": "string"},
                            "sources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source": {
                                            "type": "string",
                                            "enum": [
                                                "wikipedia",
                                                "arxiv",
                                                "pubmed",
                                                "pubmedcentral",
                                                "clinicaltrials",
                                                "wikidata",
                                                "openalex",
                                            ],
                                        },
                                        "top_k": {
                                            "type": "integer",
                                            "minimum": 5,
                                            "maximum": 50,
                                        },
                                        "confidence": {"type": "number"},
                                    },
                                    "required": ["source", "top_k", "confidence"],
                                    "additionalProperties": False,
                                },
                            },
                            "excluded_sources": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "wikipedia",
                                        "arxiv",
                                        "pubmed",
                                        "pubmedcentral",
                                        "clinicaltrials",
                                        "wikidata",
                                        "openalex",
                                    ],
                                },
                            },
                            "date_range": {
                                "type": "object",
                                "properties": {
                                    "start_date": {
                                        "type": ["string", "null"],
                                        "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                                    },
                                    "end_date": {
                                        "type": ["string", "null"],
                                        "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                                    },
                                },
                                "required": ["start_date", "end_date"],
                                "additionalProperties": False,
                            },
                            "language": {"type": ["string", "null"]},
                            "graph_entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "canonical_name": {"type": "string"}
                                    },
                                    "required": ["canonical_name"],
                                    "additionalProperties": False,
                                },
                            },
                            "graph_neighborhood": {
                                "type": "object",
                                "properties": {
                                    "hops": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 4,
                                    },
                                    "max_nodes": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 300,
                                    },
                                    "max_relationships": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 750,
                                    },
                                    "direction": {
                                        "type": "string",
                                        "enum": ["incoming", "outgoing", "both"],
                                    },
                                },
                                "required": [
                                    "hops",
                                    "max_nodes",
                                    "max_relationships",
                                    "direction",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "required": [
                            "rewritten_query",
                            "research_goal",
                            "reasoning_type",
                            "sources",
                            "excluded_sources",
                            "date_range",
                            "language",
                            "graph_entities",
                            "graph_neighborhood",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
        )
        return PlannerResponse.from_dict(json.loads(response.output_text))

    def _build_query_plan(
        self,
        original_query: str,
        planner_response: PlannerResponse,
        embedding: np.ndarray,
    ) -> QueryPlan:
        rewritten_query = planner_response.rewritten_query.strip()
        source_plans = [
            SourcePlan(
                source=source.source,
                top_k=source.top_k,
                confidence=source.confidence,
            )
            for source in planner_response.sources
        ]
        date_range = DateConstraint(
            start_date=planner_response.date_range.start_date,
            end_date=planner_response.date_range.end_date,
        )
        constraints = QueryConstraints(
            sources=[source.source for source in source_plans],
            excluded_sources=planner_response.excluded_sources,
            date_range=date_range,
            language=planner_response.language,
        )
        graph_plan = GraphRetrievalPlan(
            entities=[
                GraphEntity(canonical_name=entity.canonical_name)
                for entity in planner_response.graph_entities
            ],
            neighborhood=GraphNeighborhoodPlan(
                hops=planner_response.graph_neighborhood.hops,
                max_nodes=planner_response.graph_neighborhood.max_nodes,
                max_relationships=planner_response.graph_neighborhood.max_relationships,
                direction=planner_response.graph_neighborhood.direction,
            ),
        )
        return QueryPlan(
            query=original_query,
            rewritten_query=rewritten_query,
            embedding=embedding,
            intent=QueryIntent(
                research_goal=planner_response.research_goal,
                reasoning_type=planner_response.reasoning_type,
            ),
            constraints=constraints,
            source_plans=source_plans,
            graph_plan=graph_plan,
        )