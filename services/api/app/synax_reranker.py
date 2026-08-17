# services/api/app/synax_reranker.py
import torch
from dataclasses import dataclass
from typing import Any, Dict, List

from services.api.app.synax_config import device, reranker, reranker_tokenizer
from services.api.app.synax_hybrid_retrieval import HybridRetrievalResult
from services.api.app.synax_knowledge_graph_retrieval import KnowledgeGraphContext
from services.api.app.synax_query_planner import QueryPlan


@dataclass(slots=True)
class RankedContextItem:
    source: str
    score: float
    item_type: str
    payload: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "score": self.score,
            "item_type": self.item_type,
            "payload": self.payload.to_dict(),
        }


@dataclass(slots=True)
class RankedRetrievalContext:
    query_plan: QueryPlan
    ranked_items: List[RankedContextItem]
    knowledge_graph: KnowledgeGraphContext

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_plan": self.query_plan.to_dict(),
            "ranked_items": [item.to_dict() for item in self.ranked_items],
            "knowledge_graph": self.knowledge_graph.to_dict(),
        }


@dataclass(slots=True)
class _Candidate:
    source: str
    item_type: str
    payload: Any
    text: str


class HybridReranker:
    def __init__(self, top_k: int = 15):
        self.top_k = top_k

    @staticmethod
    def _graph_text(evidence) -> str:
        return evidence.evidence

    @staticmethod
    def _vector_text(result) -> str:
        return result.text

    def _build_candidates(
        self,
        retrieval: HybridRetrievalResult,
    ) -> List[_Candidate]:
        candidates = []

        for result in retrieval.vector_results:
            candidates.append(
                _Candidate(
                    source=result.source,
                    item_type="vector",
                    payload=result,
                    text=self._vector_text(result),
                )
            )

        for evidence in retrieval.knowledge_graph.evidence:
            candidates.append(
                _Candidate(
                    source=evidence.source,
                    item_type="graph",
                    payload=evidence,
                    text=self._graph_text(evidence),
                )
            )

        return candidates

    @torch.inference_mode()
    def rerank(
        self,
        retrieval: HybridRetrievalResult,
    ) -> RankedRetrievalContext:
        candidates = self._build_candidates(retrieval)

        if not candidates:
            return RankedRetrievalContext(
                query_plan=retrieval.query_plan,
                ranked_items=[],
                knowledge_graph=retrieval.knowledge_graph,
            )

        query = retrieval.query_plan.rewritten_query

        inputs = reranker_tokenizer(
            [[query, candidate.text] for candidate in candidates],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)

        scores = (
            reranker(**inputs)
            .logits
            .squeeze(-1)
            .detach()
            .cpu()
            .tolist()
        )

        ranked = [
            RankedContextItem(
                source=candidate.source,
                score=float(score),
                item_type=candidate.item_type,
                payload=candidate.payload,
            )
            for candidate, score in zip(candidates, scores)
        ]

        ranked.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return RankedRetrievalContext(
            query_plan=retrieval.query_plan,
            ranked_items=ranked[: self.top_k],
            knowledge_graph=retrieval.knowledge_graph,
        )