# services/api/app/synax_reasoning.py
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from enum import Enum
from textwrap import dedent
from pydantic import BaseModel
from pydantic import Field
from google.genai.types import GenerateContentConfig
from services.api.app.synax_config import gemini_client
from services.api.app.synax_knowledge_graph_retrieval import RetrievedEvidence
from services.api.app.synax_reranker import RankedRetrievalContext
from services.api.app.synax_reranker import RankedContextItem
from services.api.app.synax_research_memory import ResearchMemoryContext


class ContradictionSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class EvidenceReference:
    evidence_number: int
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_number": self.evidence_number,
            "explanation": self.explanation,
        }


@dataclass(slots=True)
class AnswerCitation:
    claim: str
    evidence: list[EvidenceReference] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass(slots=True)
class Agreement:
    summary: str
    evidence: list[EvidenceReference] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "evidence": [e.to_dict() for e in self.evidence],
            "reasoning": self.reasoning,
        }


@dataclass(slots=True)
class Contradiction:
    summary: str
    evidence: list[EvidenceReference]
    severity: ContradictionSeverity
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "evidence": [e.to_dict() for e in self.evidence],
            "severity": self.severity.value,
            "reasoning": self.reasoning,
        }


@dataclass(slots=True)
class KnowledgeGap:
    unanswered_question: str
    reasoning: str
    significance: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unanswered_question": self.unanswered_question,
            "reasoning": self.reasoning,
            "significance": self.significance,
        }


@dataclass(slots=True)
class ReasoningStep:
    step: int
    description: str
    evidence: list[EvidenceReference] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "description": self.description,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass(slots=True)
class StructuredReasoning:
    answer: str
    answer_citations: list[AnswerCitation] = field(default_factory=list)
    agreements: list[Agreement] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    knowledge_gaps: list[KnowledgeGap] = field(default_factory=list)
    reasoning_trace: list[ReasoningStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "answer_citations": [c.to_dict() for c in self.answer_citations],
            "agreements": [a.to_dict() for a in self.agreements],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "knowledge_gaps": [g.to_dict() for g in self.knowledge_gaps],
            "reasoning_trace": [r.to_dict() for r in self.reasoning_trace],
        }


class EvidenceReferenceSchema(BaseModel):
    evidence_number: int
    explanation: str = ""

    def to_domain(self) -> EvidenceReference:
        return EvidenceReference(
            evidence_number=self.evidence_number,
            explanation=self.explanation,
        )


class AnswerCitationSchema(BaseModel):
    claim: str
    evidence: list[EvidenceReferenceSchema] = Field(default_factory=list)

    def to_domain(self) -> AnswerCitation:
        return AnswerCitation(
            claim=self.claim,
            evidence=[e.to_domain() for e in self.evidence],
        )


class AgreementSchema(BaseModel):
    summary: str
    evidence: list[EvidenceReferenceSchema] = Field(default_factory=list)
    reasoning: str = ""

    def to_domain(self) -> Agreement:
        return Agreement(
            summary=self.summary,
            evidence=[e.to_domain() for e in self.evidence],
            reasoning=self.reasoning,
        )


class ContradictionSchema(BaseModel):
    summary: str
    evidence: list[EvidenceReferenceSchema]
    reasoning: str
    severity: ContradictionSeverity

    def to_domain(self) -> Contradiction:
        return Contradiction(
            summary=self.summary,
            evidence=[e.to_domain() for e in self.evidence],
            reasoning=self.reasoning,
            severity=self.severity,
        )


class KnowledgeGapSchema(BaseModel):
    unanswered_question: str
    reasoning: str
    significance: str

    def to_domain(self) -> KnowledgeGap:
        return KnowledgeGap(
            unanswered_question=self.unanswered_question,
            reasoning=self.reasoning,
            significance=self.significance,
        )


class ReasoningStepSchema(BaseModel):
    step: int
    description: str
    evidence: list[EvidenceReferenceSchema] = Field(default_factory=list)

    def to_domain(self) -> ReasoningStep:
        return ReasoningStep(
            step=self.step,
            description=self.description,
            evidence=[e.to_domain() for e in self.evidence],
        )


class StructuredReasoningSchema(BaseModel):
    answer: str
    answer_citations: list[AnswerCitationSchema] = Field(default_factory=list)
    agreements: list[AgreementSchema] = Field(default_factory=list)
    contradictions: list[ContradictionSchema] = Field(default_factory=list)
    knowledge_gaps: list[KnowledgeGapSchema] = Field(default_factory=list)
    reasoning_trace: list[ReasoningStepSchema] = Field(default_factory=list)

    def to_domain(self) -> StructuredReasoning:
        return StructuredReasoning(
            answer=self.answer,
            answer_citations=[c.to_domain() for c in self.answer_citations],
            agreements=[a.to_domain() for a in self.agreements],
            contradictions=[c.to_domain() for c in self.contradictions],
            knowledge_gaps=[g.to_domain() for g in self.knowledge_gaps],
            reasoning_trace=[r.to_domain() for r in self.reasoning_trace],
        )


@dataclass(slots=True)
class ReasoningResult:
    retrieval: RankedRetrievalContext
    reasoning: StructuredReasoning
    memory: ResearchMemoryContext

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval": self.retrieval.to_dict(),
            "reasoning": self.reasoning.to_dict(),
            "memory": self.memory.to_dict(),
        }


class GeminiReasoner:
    @staticmethod
    def _format_item(item: RankedContextItem) -> str:
        if item.item_type == "vector":
            return item.payload.text
        if item.item_type == "graph":
            evidence: RetrievedEvidence = item.payload
            return (
                f"Source entity: {evidence.source_entity}\n"
                f"Relationship: {evidence.relationship_type}\n"
                f"Target entity: {evidence.target_entity}\n\n"
                f"Evidence sentence: {evidence.evidence}"
            )
        return str(item.payload)

    @staticmethod
    def _format_memory_context(memory: ResearchMemoryContext) -> str:
        if memory.is_empty:
            return "No prior SynaX research memory was retrieved."
        return "\n\n".join(
            f"[Memory {i}]\n{item.content}"
            for i, item in enumerate(memory.memories, 1)
        )

    def _build_prompt(
        self, retrieval: RankedRetrievalContext, memory: ResearchMemoryContext
    ) -> str:
        evidence = "\n\n".join(
            f"[Evidence {i}]\n{self._format_item(item)}"
            for i, item in enumerate(retrieval.ranked_items, 1)
        )
        memory_context = self._format_memory_context(memory)
        return dedent(
            f"""You are SynaX, an AI Research Operating System. Your purpose is to help researchers understand complex information through evidence-based reasoning.

Question: {retrieval.query_plan.query}

Newly Retrieved Evidence:
{evidence}

Historical SynaX Research Memory:
{memory_context}

RESEARCH MEMORY INSTRUCTIONS:

Historical Research Memory represents the accumulated research understanding from previous analyses of this research workspace. Use it as contextual research history, not as authoritative evidence. Integrate historical research memory with the newly retrieved evidence without allowing previous conclusions to bias or predetermine your current analysis.

The newly retrieved evidence is the primary basis for the current reasoning. Research memory MUST NEVER be treated as evidence, cited as evidence, or used to establish a claim that is not supported by the newly retrieved evidence. Evidence references in the response MUST refer ONLY to the supplied newly retrieved evidence.

Use the historical memory to determine whether the current evidence:

• reinforces previously established findings;
• provides additional support for previous agreements;
• weakens or qualifies previous conclusions;
• confirms, resolves, or changes previously identified contradictions;
• answers previously identified knowledge gaps;
• partially answers and therefore refines existing knowledge gaps;
• reveals new agreements, contradictions, or knowledge gaps that were not previously identified.

Previous answers are historical conclusions. DO NOT simply repeat or copy them. Re-evaluate their claims against the newly retrieved evidence and produce the best current understanding.

Previous agreements represent findings that previously appeared to be supported by converging evidence. Preserve them ONLY when the newly retrieved evidence continues to support them. If the new evidence strengthens, narrows, qualifies, or challenges an agreement, REFLECT that updated understanding.

Previous contradictions represent conflicts identified during earlier research. Determine whether the newly retrieved evidence confirms, resolves, weakens, or changes those conflicts. DO NOT reproduce a previous contradiction merely because it exists in memory.

Previous knowledge gaps represent unresolved research questions from earlier analyses. Determine whether the newly retrieved evidence resolves them, partially addresses them, leaves them unresolved, or reveals a more precise remaining question. DO NOT report a previous knowledge gap if the current evidence adequately answers it.

When historical research memory conflicts with newly retrieved evidence, prioritize the newly retrieved evidence. DO NOT assume that a previous conclusion remains correct correct simply because it appears in memory.

DO NOT manufacture continuity. If the current evidence does not meaningfully relate to a previous memory, ignore that memory rather than forcing it into the analysis.

Maintain uncertainty across research iterations. A previous conclusion that was uncertain MUST NOT become certain merely because it is stored in memory. Likewise, DO NOT weaken a previously well-supported conclusion without evidence supporting the change.

The current answer must represent the updated state of understanding after considering BOTH the newly retrieved evidence and historical research memory. The purpose of historical research memory is to preserve continuity across research sessions, allowing the current analysis to build upon, reassess, and update previously established research understanding rather from starting from scratch.


Your task is to carefully analyze the supplied evidence and return a JSON object that conforms exactly to the provided response schema.


Field Requirements:

answer:
The answer MUST consist entirely of fully developed, long-form paragraphs. Never use bullet points, numbered lists, fragments, headings, tables, or short chatbot-style responses.

The depth, length, and number of paragraphs MUST scale according to the complexity, scope, ambiguity, and evidentiary demands of the research question and the supplied evidence.

For a focused question with a clear and well-supported answer, provide a concise long-form analysis that fully addresses the question without unnecessary expansion.

For a complex, multidisciplinary, ambiguous, or evidence-rich question, provide a substantially more extensive long-form analysis as supported by the evidence.

DO NOT artificially shorten an answer merely to be concise, and DO NOT artificially lengthen an answer merely to satisfy a perception of thoroughness. EVERY paragraph must contribute meaningful analytical value.

The answer MUST be information-dense, analytically rigorous, logically connected, and written with the depth and precision of an experienced multidisciplinary researcher. Communicate enough detail to develop an accurate and deep understanding of the research question while remaining proportionate to what the question actually requires.

The answer MUST remain STRICTLY paragraph-based regardless of question complexity. Even when the answer contains multiple distinct findings or dimensions, integrate them into coherent, logically connected paragraphs rather than converting them into lists or other structured formats.

answer_citations:
Map specific claims or sentences in the answer to the newly supplied evidence that directly supports them. For each answer citation:

• claim MUST contain the exact sentence or contiguous text from the answer that the citation supports.
• evidence MUST contain ONLY evidence references from the supplied newly retrieved evidence.
• Every evidence number MUST correspond to an actual supplied evidence item.
• Use multiple evidence references when a claim is supported by multiple evidence items.
• NEVER cite historical research memory.
• Every substantive factual claim in the answer that depends on supplied evidence MUST have an answer citation.
• The claim text MUST match the corresponding text in "answer" EXACTLY.

agreements:
Identify important findings that are consistently supported by multiple supplied evidence items. For each agreement:

• Summary: State the central finding or conclusion supported by the converging evidence.
• Evidence: Reference the evidence items that directly support the agreement and explain how each referenced evidence item contributes to it.
• Reasoning: Explain why the referenced evidence collectively supports the agreement and how the findings converge.

contradictions:
Identify genuine conflicts or inconsistencies between the supplied evidence. Only include actual contradictions. DO NOT treat differences in wording, emphasis, study context, or level of detail as contradictions unless they produce substantively conflicting conclusions. For each contradiction:

• Summary: State the central conflict between the evidence items.
• Evidence: Reference the evidence items that directly participate in the contradiction and explain how each referenced evidence item contributes to the conflict.
• Reasoning: Explain why the referenced evidence conflicts and, where supported by the supplied evidence, identify the factors that may account for the disagreement.
• Severity: Assign an appropriate severity level based on how substantially the contradiction affects the reliability or interpretation of the overall conclusion. Use only: low, moderate, high, or critical.

Only report contradictions that are supported by the supplied evidence. Do not manufacture or infer conflicts that are not present in the evidence.

knowledge_gaps:
Identify important unanswered research questions that emerge from the supplied evidence. For each knowledge gap:

• Unanswered Question: State as a clear, specific research question.
• Reasoning: Explain what is missing or too limited in the supplied evidence to answer it.
• Significance: Briefly state why answering this would matter.

reasoning_trace:
Describe the major evidence-based reasoning steps used to derive the final answer. For each reasoning step:

• Step: Provide the sequential step number.
• Description: Describe the major analytical inference or synthesis represented by this step.
• Evidence: Reference the evidence items that directly support this step and explain how each referenced evidence item contributes to the step.

The reasoning trace must summarize the observable evidence synthesis and MUST NOT reveal internal reasoning, hidden deliberation, or chain-of-thought.

General Requirements:

- Return ONLY the JSON object.
- Do not return Markdown.
- Do not return explanations outside the JSON.
- Never invent evidence numbers.
- Never invent evidence.
- Never invent relationships.
- Never reference evidence that was not supplied.
- If no agreements exist, return an empty agreements array.
- If no contradictions exist, return an empty contradictions array.
- If no knowledge gaps exist, return an empty knowledge_gaps array.
- If no reasoning steps are appropriate, return an empty reasoning_trace array."""
        )

    async def reason(
        self,
        retrieval: RankedRetrievalContext,
        memory: ResearchMemoryContext,
    ) -> ReasoningResult:
        response = await gemini_client.aio.models.generate_content(
            model="gemini-3.7-flash",
            contents=self._build_prompt(retrieval, memory),
            config=GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StructuredReasoningSchema,
            ),
        )
        if response.parsed is None:
            raise ValueError("Gemini returned no structured reasoning.")
        return ReasoningResult(
            retrieval=retrieval,
            reasoning=response.parsed.to_domain(),
            memory=memory,
        )