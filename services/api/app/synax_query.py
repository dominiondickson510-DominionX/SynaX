# services/api/app/synax_query.py
import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.app.synax_billing import (
    QUERY_CREDIT_COST,
    consume_credits,
    refund_credits,
)
from services.api.app.synax_citation_generator import CitationGenerator
from services.api.app.synax_config import embedder
from services.api.app.synax_hybrid_retrieval import HybridRetriever
from services.api.app.synax_observability import log_event
from services.api.app.synax_reasoning import GeminiReasoner
from services.api.app.synax_research_memory import (
    SynaXResearchMemory,
    SupermemoryResearchMemory,
)
from services.api.app.synax_research_workspaces import (
    ResearchHistory,
    Workspace,
    get_current_user,
    get_session,
)
from services.api.app.synax_reranker import HybridReranker


retriever = HybridRetriever(embedder)
reranker = HybridReranker()
research_memory = SynaXResearchMemory(SupermemoryResearchMemory())
reasoner = GeminiReasoner()
citation_generator = CitationGenerator()


class ResearchQuery(BaseModel):
    query: str
    workspace_id: str


async def process_query(*, query: str, workspace_id: str):
    hybrid_result = await retriever.retrieve(query)

    memory_context, ranked_context = await asyncio.gather(
        research_memory.retrieve(
            workspace_id=workspace_id,
            query=hybrid_result.query_plan.rewritten_query,
            limit=10,
        ),
        asyncio.to_thread(reranker.rerank, hybrid_result),
    )

    reasoning_result = await reasoner.reason(
        ranked_context,
        memory_context,
    )

    return await citation_generator.generate(reasoning_result)


router = APIRouter(
    prefix="/synax",
    tags=["Research"],
)


@router.post("/query")
async def query(
    request: ResearchQuery,
    user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    workspace = (
        await session.execute(
            select(Workspace).where(
                Workspace.id == request.workspace_id,
                Workspace.user_id == user["user_id"],
            )
        )
    ).scalar_one_or_none()

    if workspace is None:
        log_event(
            "research_query_rejected",
            status="failed",
            user_id=user["user_id"],
            workspace_id=request.workspace_id,
            reason="workspace_not_found",
        )
        raise HTTPException(
            status_code=404,
            detail="Workspace not found",
        )

    log_event(
        "research_query_started",
        status="started",
        user_id=user["user_id"],
        workspace_id=workspace.id,
    )

    await consume_credits(
        user_id=user["user_id"],
        cost=QUERY_CREDIT_COST,
        description=f"Research query in workspace {workspace.id}",
        session=session,
    )
    await session.commit()

    try:
        reasoning_result = await process_query(
            query=request.query,
            workspace_id=workspace.id,
        )
    except Exception as exc:
        await session.rollback()

        await refund_credits(
            user_id=user["user_id"],
            amount=QUERY_CREDIT_COST,
            description=f"Refund for failed research query in workspace {workspace.id}",
            session=session,
        )
        await session.commit()

        log_event(
            "research_query_failed",
            status="failed",
            user_id=user["user_id"],
            workspace_id=workspace.id,
            error=str(exc),
        )

        raise

    history_id = str(uuid.uuid4())

    history = ResearchHistory(
        id=history_id,
        workspace_id=workspace.id,
        query=request.query,
        result=reasoning_result.to_dict(),
        memory_sync_status="pending",
    )

    session.add(history)
    await session.commit()
    await session.refresh(history)

    memory_id = f"synax_research_{history_id}"

    try:
        memory_result = await research_memory.remember_reasoning(
            workspace_id=workspace.id,
            memory_id=memory_id,
            research_history_id=history_id,
            query=request.query,
            reasoning=reasoning_result.reasoning,
        )

        history.memory_sync_status = "synced"
        history.memory_provider = memory_result.provider
        history.memory_id = memory_id
        history.memory_sync_error = None
        history.memory_synced_at = datetime.now(timezone.utc)

    except Exception as exc:
        history.memory_sync_status = "failed"
        history.memory_sync_error = str(exc)

        log_event(
            "research_memory_sync_failed",
            status="failed",
            user_id=user["user_id"],
            workspace_id=workspace.id,
            history_id=history_id,
            error=str(exc),
        )

    await session.commit()

    log_event(
        "research_query_completed",
        status="success",
        user_id=user["user_id"],
        workspace_id=workspace.id,
        history_id=history_id,
    )

    return {
        "workspace_id": workspace.id,
        "history_id": history_id,
        "credits_consumed": float(QUERY_CREDIT_COST),
        "reasoning_result": reasoning_result.to_dict(),
    }