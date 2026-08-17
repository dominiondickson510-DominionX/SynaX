# services/api/app/synax_research_memory.py
import json
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from services.api.app.synax_config import supermemory_client


@dataclass(slots=True)
class ResearchMemory:
    memory_id: Optional[str]
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
        }


@dataclass(slots=True)
class ResearchMemoryContext:
    memories: List[ResearchMemory] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memories": [memory.to_dict() for memory in self.memories]
        }

    @property
    def is_empty(self) -> bool:
        return not self.memories


class ResearchMemoryPayload(Protocol):
    answer: str
    agreements: list[Any]
    contradictions: list[Any]
    knowledge_gaps: list[Any]


@dataclass(slots=True)
class AddMemoryResponse:
    provider: str
    history_id: Optional[str]
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "history_id": self.history_id,
            "status": self.status,
        }


class ResearchMemoryProvider(ABC):
    @abstractmethod
    async def retrieve(
        self,
        *,
        workspace_id: str,
        query: str,
        limit: int = 10,
    ) -> ResearchMemoryContext:
        raise NotImplementedError

    @abstractmethod
    async def remember(
        self,
        *,
        workspace_id: str,
        memory_id: str,
        content: str,
        history_id: str,
    ) -> AddMemoryResponse:
        raise NotImplementedError


class SupermemoryResearchMemory(ResearchMemoryProvider):
    provider_name = "supermemory"

    def __init__(self):
        self.client = supermemory_client()

    @staticmethod
    def _container_tag(workspace_id: str) -> str:
        return f"synax_research_workspace_{workspace_id}"

    async def retrieve(
        self,
        *,
        workspace_id: str,
        query: str,
        limit: int = 10,
    ) -> ResearchMemoryContext:
        if not query.strip():
            return ResearchMemoryContext()
        response = await self.client.search.memories(
            q=query,
            search_mode="memories",
            container_tag=self._container_tag(workspace_id),
            limit=limit,
        )
        memories: List[ResearchMemory] = []
        for result in response.results:
            memories.append(
                ResearchMemory(
                    memory_id=result.custom_id,
                    content=result.content,
                )
            )
        return ResearchMemoryContext(memories=memories)

    async def remember(
        self,
        *,
        workspace_id: str,
        memory_id: str,
        content: str,
        history_id: str,
    ) -> AddMemoryResponse:
        response = await self.client.add(
            content=content,
            custom_id=memory_id,
            container_tag=self._container_tag(workspace_id),
        )
        return AddMemoryResponse(
            provider=self.provider_name,
            history_id=history_id,
            status=response.status,
        )


class SynaXResearchMemory:
    def __init__(self, provider: ResearchMemoryProvider):
        self.provider = provider

    async def retrieve(
        self,
        *,
        workspace_id: str,
        query: str,
        limit: int = 10,
    ) -> ResearchMemoryContext:
        memories = await self.provider.retrieve(
            workspace_id=workspace_id,
            query=query,
            limit=limit,
        )
        if memories.is_empty:
            return ResearchMemoryContext()
        return memories

    @staticmethod
    def _build_memory_content(
        *,
        query: str,
        reasoning: ResearchMemoryPayload,
    ) -> str:
        memory_payload = {
            "query": query,
            "answer": reasoning.answer,
            "agreements": [
                {
                    "summary": agreement.summary,
                    "reasoning": agreement.reasoning,
                }
                for agreement in reasoning.agreements
            ],
            "contradictions": [
                {
                    "summary": contradiction.summary,
                    "reasoning": contradiction.reasoning,
                }
                for contradiction in reasoning.contradictions
            ],
            "knowledge_gaps": [
                {
                    "unanswered_question": gap.unanswered_question,
                    "reasoning": gap.reasoning,
                    "significance": gap.significance,
                }
                for gap in reasoning.knowledge_gaps
            ],
        }
        return json.dumps(memory_payload, ensure_ascii=False)

    async def remember_reasoning(
        self,
        *,
        workspace_id: str,
        memory_id: str,
        research_history_id: str,
        query: str,
        reasoning: ResearchMemoryPayload,
    ) -> AddMemoryResponse:
        content = self._build_memory_content(
            query=query,
            reasoning=reasoning,
        )
        return await self.provider.remember(
            workspace_id=workspace_id,
            memory_id=memory_id,
            content=content,
            history_id=research_history_id,
        )