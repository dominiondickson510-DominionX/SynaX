# services/api/app/synax_citation_generator.py
import httpx
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from services.api.app.synax_reasoning import StructuredReasoning
from services.api.app.synax_reasoning import ReasoningResult
from services.api.app.synax_reasoning import EvidenceReference
from services.api.app.synax_reranker import RankedRetrievalContext
from services.api.app.synax_reranker import RankedContextItem


@dataclass(slots=True)
class Citation:
    citation_id: str
    evidence_number: int
    document_id: Optional[str]
    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "evidence_number": self.evidence_number,
            "document_id": self.document_id,
            "source": self.source,
            "text": self.text,
            "metadata": self.metadata,
            "provenance": self.provenance,
        }


@dataclass(slots=True)
class CitationSet:
    citations: List[Citation] = field(default_factory=list)
    formatted: Optional[str] = None
    style: str = "apa"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citations": [citation.to_dict() for citation in self.citations],
            "formatted": self.formatted,
            "style": self.style,
        }


@dataclass(slots=True)
class CitationResult:
    retrieval: RankedRetrievalContext
    reasoning: StructuredReasoning
    citations: CitationSet

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval": self.retrieval.to_dict(),
            "reasoning": self.reasoning.to_dict(),
            "citations": self.citations.to_dict(),
        }


class CitationGenerator:
    @staticmethod
    def _extract_citation_context(
        ranked_items: List[RankedContextItem],
    ) -> Dict[int, Citation]:
        citations = {}
        for evidence_number, item in enumerate(ranked_items, 1):
            payload = item.payload
            if item.item_type == "vector":
                metadata = dict(getattr(payload, "metadata", {}) or {})
                citations[evidence_number] = Citation(
                    citation_id=f"E{evidence_number}",
                    evidence_number=evidence_number,
                    document_id=getattr(payload, "document_id", None),
                    source=getattr(payload, "source", ""),
                    text=metadata.get("text", getattr(payload, "text", "")),
                    metadata=metadata,
                    provenance=list(metadata.get("provenance", [])),
                )
            elif item.item_type == "graph":
                evidence = getattr(payload, "evidence", "")
                citations[evidence_number] = Citation(
                    citation_id=f"E{evidence_number}",
                    evidence_number=evidence_number,
                    document_id=None,
                    source=getattr(payload, "source", ""),
                    text=evidence,
                    metadata={},
                    provenance=[],
                )
        return citations

    @staticmethod
    def _resolve_reference(
        evidence_number: int, citation_index: Dict[int, Citation]
    ) -> Optional[Citation]:
        return citation_index.get(evidence_number)

    @staticmethod
    def _inject_citations(
        answer: str,
        reasoning: StructuredReasoning,
        citation_index: Dict[int, Citation],
    ) -> str:
        injections = []
        occupied = []
        for answer_citation in reasoning.answer_citations:
            claim = answer_citation.claim.strip()
            if not claim:
                continue
            evidence_numbers = sorted(
                {
                    reference.evidence_number
                    for reference in answer_citation.evidence
                    if reference.evidence_number in citation_index
                }
            )
            if not evidence_numbers:
                continue
            start = answer.find(claim)
            if start < 0:
                continue
            end = start + len(claim)
            if any(
                start < existing_end and end > existing_start
                for existing_start, existing_end in occupied
            ):
                continue
            injections.append(
                (
                    start,
                    end,
                    " " + " ".join(f"[E{n}]" for n in evidence_numbers),
                )
            )
            occupied.append((start, end))
        for start, end, marker in sorted(injections, key=lambda item: item[0], reverse=True):
            answer = answer[:end] + marker + answer[end:]
        return answer

    @staticmethod
    def _metadata_to_csl(citation: Citation) -> Dict[str, Any]:
        metadata = dict(citation.metadata)
        authors = metadata.get("authors", [])
        normalized_authors = []
        if isinstance(authors, str):
            authors = [authors]
        for author in authors:
            if isinstance(author, dict):
                normalized_authors.append(author)
                continue
            if not isinstance(author, str):
                continue
            author = author.strip()
            if not author:
                continue
            if "," in author:
                family, given = author.split(",", 1)
                normalized_authors.append(
                    {"family": family.strip(), "given": given.strip()}
                )
            else:
                parts = author.split()
                if len(parts) == 1:
                    normalized_authors.append({"family": parts[0]})
                else:
                    normalized_authors.append(
                        {"given": " ".join(parts[:-1]), "family": parts[-1]}
                    )
        publication_date = metadata.get("publication_date")
        issued = None
        if publication_date:
            if isinstance(publication_date, str):
                parts = publication_date[:10].split("-")
                try:
                    issued = {"date-parts": [[int(part) for part in parts]]}
                except ValueError:
                    issued = None
            elif isinstance(publication_date, (int, float)):
                issued = {"date-parts": [[int(publication_date)]]}
        title = metadata.get("title", "")
        journal = metadata.get("journal") or metadata.get("journal_ref")
        doi = metadata.get("doi")
        url = metadata.get("url")
        csl = {"title": title, "author": normalized_authors}
        if issued:
            csl["issued"] = issued
        if journal:
            csl["container-title"] = journal
        if doi:
            csl["DOI"] = doi
        if url:
            csl["URL"] = url
        if metadata.get("pmid"):
            csl["PMID"] = metadata["pmid"]
        if metadata.get("pmcid"):
            csl["PMCID"] = metadata["pmcid"]
        if metadata.get("nct_id"):
            csl["NCT-ID"] = metadata["nct_id"]
        if metadata.get("entry_id"):
            csl["archive_location"] = metadata["entry_id"]
        if metadata.get("pdf_url"):
            csl.setdefault("URL", metadata["pdf_url"])
        if metadata.get("page_id"):
            csl["archive_location"] = metadata["page_id"]
        if metadata.get("abstract"):
            csl["abstract"] = metadata["abstract"]
        if metadata.get("journal") or metadata.get("journal_ref"):
            csl["type"] = "article-journal"
        elif metadata.get("nct_id"):
            csl["type"] = "article"
        elif metadata.get("pdf_url"):
            csl["type"] = "article"
        elif metadata.get("url"):
            csl["type"] = "webpage"
        else:
            csl["type"] = "document"
        return csl

    async def _format(
        self,
        citations: List[Citation],
        style: str = "apa",
        output_type: str = "bibliography",
        format: str = "text",
        lang: str = "en-US",
    ) -> str:
        items = [self._metadata_to_csl(citation) for citation in citations]
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://127.0.0.1:3100/format",
                json={
                    "items": items,
                    "style": style,
                    "output_type": output_type,
                    "format": format,
                    "lang": lang,
                },
            )
        response.raise_for_status()
        data = response.json()
        return data["result"]

    async def generate(
        self,
        reasoning_result: ReasoningResult,
        style: str = "apa",
        output_type: str = "bibliography",
        format: str = "text",
        lang: str = "en-US",
    ) -> CitationResult:
        citation_index = self._extract_citation_context(
            reasoning_result.retrieval.ranked_items
        )
        references = []
        for answer_citation in reasoning_result.reasoning.answer_citations:
            references.extend(
                reference
                for reference in answer_citation.evidence
                if reference.evidence_number in citation_index
            )
        for agreement in reasoning_result.reasoning.agreements:
            references.extend(
                reference
                for reference in agreement.evidence
                if reference.evidence_number in citation_index
            )
        for contradiction in reasoning_result.reasoning.contradictions:
            references.extend(
                reference
                for reference in contradiction.evidence
                if reference.evidence_number in citation_index
            )
        citations = []
        seen = set()
        for reference in references:
            evidence_number = reference.evidence_number
            if evidence_number in seen:
                continue
            seen.add(evidence_number)
            citation = self._resolve_reference(
                evidence_number=evidence_number, citation_index=citation_index
            )
            if citation is not None:
                citations.append(citation)
        citations.sort(key=lambda citation: citation.evidence_number)
        formatted = await self._format(
            citations=citations,
            style=style,
            output_type=output_type,
            format=format,
            lang=lang,
        )
        reasoning_result.reasoning.answer = self._inject_citations(
            answer=reasoning_result.reasoning.answer,
            reasoning=reasoning_result.reasoning,
            citation_index=citation_index,
        )
        return CitationResult(
            retrieval=reasoning_result.retrieval,
            reasoning=reasoning_result.reasoning,
            citations=CitationSet(citations=citations, formatted=formatted, style=style),
        )