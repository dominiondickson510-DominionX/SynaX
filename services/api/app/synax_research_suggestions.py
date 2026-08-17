# services/api/app/synax_research_suggestions.py
from pydantic import BaseModel,Field
from fastapi import APIRouter,Depends,HTTPException
from google.genai.types import GenerateContentConfig
from services.api.app.synax_config import gemini_client
from services.api.app.synax_research_workspaces import get_current_user,get_session,Workspace
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

class ResearchQuestionSuggestions(BaseModel):
    suggestions:list[str]=Field(min_length=12,max_length=12)

class ResearchSuggestionsResponse(BaseModel):
    workspace_id:str;suggestions:list[str]

router=APIRouter(prefix="/synax",tags=["Research Suggestions"])
@router.post("/workspace/{workspace_id}/suggestions",response_model=ResearchSuggestionsResponse)
async def generate_research_suggestions(workspace_id:str,user=Depends(get_current_user),session:AsyncSession=Depends(get_session)):
    workspace=(await session.execute(select(Workspace).where(Workspace.id==workspace_id,Workspace.user_id==user["user_id"]))).scalar_one_or_none()
    if workspace is None:raise HTTPException(status_code=404,detail="Workspace not found")
    title=workspace.title.strip()
    if not title:raise HTTPException(status_code=400,detail="Workspace title is required")
    prompt=f"""
You are the research-direction agent for SynaX, an AI Research Operating System designed for researchers investigating complex questions.

A researcher has just created a research workspace.

Workspace topic:
{title}

Your task is to generate EXACTLY 12 sophisticated research questions that could serve as strong starting points for serious investigation within this workspace.

These are NOT beginner questions, educational prompts, definitions, trivia questions, or generic questions about the topic.

The questions MUST be sufficiently complex that a researcher would plausibly want SynaX to investigate them rather than answer them from general knowledge.

The questions MUST remain strictly grounded in the workspace topic.

Where appropriate, formulate questions around dimensions such as:

- competing explanations or hypotheses;
- causal mechanisms and pathways;
- relationships between important entities, variables, or processes;
- effectiveness, comparative effectiveness, or limitations of interventions;
- conflicting or heterogeneous findings across studies;
- evidence supporting or weakening established assumptions;
- methodological limitations and sources of uncertainty;
- factors explaining divergent outcomes;
- translational or real-world implications;
- emerging developments that could change current understanding;
- unresolved questions and important knowledge gaps;
- interactions between multiple mechanisms, interventions, or factors.

DO NOT force these dimensions when they are inappropriate for the workspace topic. Select the dimensions that produce the MOST intellectually valuable questions.

The twelve questions MUST explore substantially different research directions. DO NOT generate twelve variations of the same question.

DO NOT answer the questions.

DO NOT number the questions.

DO NOT mention SynaX.

Each suggestion must be a complete, precise research question that can be submitted directly as a research query.

The questions MUST reflect the kind of difficult, evidence-intensive investigation for which an AI Research Operating System is useful.

Return ONLY the JSON object conforming exactly to the supplied response schema.
"""
    response=await gemini_client.aio.models.generate_content(model="gemini-3.5-flash-lite",contents=prompt,config=GenerateContentConfig(response_mime_type="application/json",response_schema=ResearchQuestionSuggestions))
    if response.parsed is None:raise HTTPException(status_code=502,detail="Research question suggestions could not be generated")
    return{"workspace_id":workspace_id,"suggestions":response.parsed.suggestions}