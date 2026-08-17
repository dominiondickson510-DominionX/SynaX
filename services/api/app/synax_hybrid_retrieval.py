# services/api/app/synax_hybrid_retrieval.py
import asyncio
from dataclasses import dataclass
from typing import List

from services.api.app.synax_query_planner import GPTQueryPlanner, QueryPlan
from services.api.app.synax_knowledge_graph_retrieval import (
    EntityRequest,
    NeighborhoodRequest,
    RelationshipDirection,
    RetrievalRequest,
    KnowledgeGraphContext,
    KnowledgeGraphRetrieval,
)
from services.api.app.synax_faiss_retrieval import (
    FaissSearchResult,
    MultiSourceFaissRetriever,
)


@dataclass(slots=True)
class HybridRetrievalResult:
    query_plan: QueryPlan
    knowledge_graph: KnowledgeGraphContext
    vector_results: List[FaissSearchResult]


class HybridRetriever:
    def __init__(self, embedder):
        self.query_planner = GPTQueryPlanner(embedder)
        self.knowledge_graph = KnowledgeGraphRetrieval()
        self.vector_retriever = MultiSourceFaissRetriever()

    async def retrieve(self, query: str) -> HybridRetrievalResult:
        query_plan = await self.query_planner.plan(query)

        graph_request = RetrievalRequest(
            entities=[
                EntityRequest(canonical_name=entity.canonical_name)
                for entity in query_plan.graph_plan.entities
            ],
            neighborhood=NeighborhoodRequest(
                hops=query_plan.graph_plan.neighborhood.hops,
                max_nodes=query_plan.graph_plan.neighborhood.max_nodes,
                max_relationships=query_plan.graph_plan.neighborhood.max_relationships,
                direction=RelationshipDirection(
                    query_plan.graph_plan.neighborhood.direction
                ),
            ),
        )

        vector_top_k = sum(
            source.top_k for source in query_plan.source_plans
        )

        knowledge_graph_task = asyncio.to_thread(
            self.knowledge_graph.retrieve,
            graph_request,
        )

        vector_retrieval_task = asyncio.to_thread(
            self.vector_retriever.search,
            query_plan.embedding,
            query_plan.source_plans,
            vector_top_k,
        )

        knowledge_graph, vector_results = await asyncio.gather(
            knowledge_graph_task,
            vector_retrieval_task,
        )

        return HybridRetrievalResult(
            query_plan=query_plan,
            knowledge_graph=knowledge_graph,
            vector_results=vector_results,
        )