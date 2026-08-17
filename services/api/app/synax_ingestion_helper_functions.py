# services/api/app/synax_ingestion_helper_functions.py
import os
import re
import json
import html
import hashlib
import unicodedata
import fitz
import redis.asyncio as redis
from threading import Lock
from bs4 import BeautifulSoup
from services.api.app.synax_config import (
    DATA_DIR,
    WIKI_LANG,
    WIKIDATA_CACHE_PATH,
    REDIS_URL,
    INGESTION_ENABLED_KEY,
)

try:
    with open(WIKIDATA_CACHE_PATH, "r", encoding="utf-8") as f:
        WIKIDATA_LABEL_CACHE = json.load(f)
except Exception:
    WIKIDATA_LABEL_CACHE = {}

PMC_SECTION_TAXONOMY = {
    "introduction": [
        "introduction",
        "intro",
        "background",
        "overview",
        "motivation",
        "aim",
        "aims",
        "objective",
        "objectives",
        "purpose",
    ],
    "methods": [
        "methods",
        "method",
        "materials and methods",
        "materials & methods",
        "methodology",
        "experimental methods",
        "experimental procedure",
        "experimental procedures",
        "study design",
        "patients and methods",
        "statistical analysis",
    ],
    "results": [
        "results",
        "findings",
        "observations",
        "experimental results",
        "evaluation",
        "performance evaluation",
        "analysis",
    ],
    "discussion": [
        "discussion",
        "general discussion",
        "interpretation",
        "limitations",
        "future work",
        "future directions",
        "implications",
    ],
    "conclusion": [
        "conclusion",
        "conclusions",
        "summary",
        "summary and conclusions",
        "summary and conclusion",
        "concluding remarks",
        "final remarks",
        "closing remarks",
    ],
    "related_work": [
        "related work",
        "previous work",
        "prior work",
        "literature review",
        "review of literature",
    ],
    "acknowledgement": ["acknowledgement", "acknowledgments"],
    "funding": ["funding", "financial support", "grant support"],
    "supplementary": [
        "supplementary material",
        "supplementary materials",
        "appendix",
        "appendices",
        "supporting information",
    ],
}

MONTH_MAP = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}


def save_text(text, domain, filename):
    path = os.path.join(DATA_DIR, domain, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def clean_text(text):
    return re.sub(r"\s+", " ", re.sub(r"<.*?>", "", text)).strip()


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", name).strip("_")


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    words = [""] * (max(pos for positions in inverted_index.values() for pos in positions) + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


def save_wikidata_cache():
    temp = WIKIDATA_CACHE_PATH + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(WIKIDATA_LABEL_CACHE, f, ensure_ascii=False, indent=2)
    os.replace(temp, WIKIDATA_CACHE_PATH)


async def batch_resolve_wikidata_labels(ids, client, batch_size=50):
    unresolved = [wid for wid in ids if wid not in WIKIDATA_LABEL_CACHE]
    for i in range(0, len(unresolved), batch_size):
        batch = unresolved[i : i + batch_size]
        try:
            response = await client.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": "|".join(batch),
                    "languages": "en",
                    "props": "labels",
                },
            )
            response.raise_for_status()
            entities = response.json().get("entities", {})
            for wid, data in entities.items():
                WIKIDATA_LABEL_CACHE[wid] = (
                    data.get("labels", {}).get("en", {}).get("value", wid)
                )
        except Exception as e:
            print(f"[Wikidata Batch Error] {batch[:3]}... : {e}")
    save_wikidata_cache()
    return {wid: WIKIDATA_LABEL_CACHE.get(wid, wid) for wid in ids}


async def batch_fetch_wikidata_entities(qids, client, batch_size=50):
    entities = {}
    for i in range(0, len(qids), batch_size):
        batch = qids[i : i + batch_size]
        try:
            response = await client.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": "|".join(batch),
                    "languages": "|".join(WIKI_LANG),
                    "props": "labels|descriptions|aliases|claims",
                },
            )
            response.raise_for_status()
            entities.update(response.json().get("entities", {}))
        except Exception as e:
            print(f"[Wikidata Batch Fetch Error] {batch[:3]}... : {e}")
    return entities


def normalize_section_name(text: str) -> str:
    if not text:
        return "other"
    heading = text.lower()
    heading = re.sub(r"^[0-9ivxIVX().\-\s]+", "", heading)
    heading = re.sub(r"[^a-z0-9 ]", " ", heading)
    heading = re.sub(r"\s+", " ", heading).strip()
    for section_type, synonyms in PMC_SECTION_TAXONOMY.items():
        if heading in synonyms:
            return section_type
    for section_type, synonyms in PMC_SECTION_TAXONOMY.items():
        if any(s in heading for s in synonyms):
            return section_type
    return "other"


def extract_text(node):
    if node is None:
        return ""
    text = node.get_text(" ", strip=True)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_references(soup):
    refs = []
    for ref in soup.find_all("ref"):
        refs.append({"id": ref.get("id", ""), "text": extract_text(ref)})
    return refs


def parse_figure(fig):
    fig_id = fig.get("id", "")
    caption_node = fig.find("caption")
    caption_text = []
    if caption_node:
        for p in caption_node.find_all(["title", "p"], recursive=True):
            text = extract_text(p)
            if text:
                caption_text.append(text)
    graphics = []
    for graphic in fig.find_all("graphic"):
        href = graphic.get("{http://www.w3.org/1999/xlink}href", "")
        graphics.append({"type": "image", "src": href})
    return {"type": "figure", "id": fig_id, "caption": caption_text, "images": graphics}


def parse_table(table):
    table_id = table.get("id", "")
    caption = extract_text(table.find("caption"))
    header = []
    rows = []
    for tr in table.find_all("tr"):
        header_cells = tr.find_all("th")
        data_cells = tr.find_all("td")
        if header_cells and not header:
            header = [extract_text(th) for th in header_cells]
            continue
        if data_cells:
            row = [extract_text(td) for td in data_cells]
            if header and len(row) == len(header):
                rows.append(dict(zip(header, row)))
            else:
                rows.append(row)
    return {"type": "table", "id": table_id, "caption": caption, "columns": header, "rows": rows}


def parse_equation(eq):
    def extract_formula_content(node):
        mathml = node.find("math")
        if mathml:
            return extract_text(mathml)
        for tag in ["tex-math", "mml:math"]:
            el = node.find(tag)
            if el:
                return extract_text(el)
        return extract_text(node)

    if eq.name == "disp-formula":
        return {
            "type": "display_equation",
            "id": eq.get("id", ""),
            "label": extract_text(eq.find("label")),
            "latex": extract_formula_content(eq),
        }
    return {"type": "inline_equation", "id": eq.get("id", ""), "latex": extract_formula_content(eq)}


def parse_section(sec, level=1, parent_id=None):
    section = {
        "id": sec.get("id", ""),
        "parent_id": parent_id,
        "level": level,
        "heading": extract_text(sec.find("title", recursive=False)),
        "type": normalize_section_name(extract_text(sec.find("title", recursive=False))),
        "content": [],
        "children": [],
    }
    parsers = {
        "p": lambda n: {"type": "paragraph", "text": extract_text(n)},
        "fig": parse_figure,
        "table-wrap": parse_table,
        "disp-formula": parse_equation,
        "inline-formula": parse_equation,
    }
    for child in sec.children:
        if getattr(child, "name", None) is None:
            continue
        if child.name == "title":
            continue
        if child.name == "sec":
            section["children"].append(parse_section(child, level + 1, section["id"]))
            continue
        parser = parsers.get(child.name)
        if parser is None:
            continue
        result = parser(child)
        if isinstance(result, list):
            section["content"].extend(result)
        else:
            section["content"].append(result)
    return section


def extract_sections(soup):
    body = soup.find("body")
    if body is None:
        return []
    return [parse_section(sec) for sec in body.find_all("sec", recursive=False)]


def extract_pmc_structure(xml_text: str, pmid: str, pmcid: str, domain: str):
    soup = BeautifulSoup(xml_text, "xml")
    structured = {
        "source": "pubmedcentral",
        "domain": domain,
        "pmid": pmid,
        "pmcid": pmcid,
        "title": extract_text(soup.find("article-title")),
        "abstract": extract_text(soup.find("abstract")),
        "sections": extract_sections(soup),
        "references": parse_references(soup),
        "acknowledgements": extract_text(soup.find("ack")),
        "authors": [
            extract_text(a) for a in soup.find_all("contrib") if extract_text(a)
        ],
    }
    return structured


async def extract_pdf_text(client, pdf_url, semaphore):
    if not pdf_url:
        return ""
    async with semaphore:
        try:
            response = await client.get(pdf_url)
            response.raise_for_status()
            pdf = fitz.open(stream=response.content, filetype="pdf")
            pages = []
            for page in pdf:
                text = page.get_text("text").strip()
                if text:
                    pages.append(text)
            pdf.close()
            return "\n\n".join(pages)
        except Exception as e:
            print(f"[PDF Error] {pdf_url}: {e}")
            return ""


def _ct_get(data, *keys, default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _ct_to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v]
    return [value]


def _ct_join(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v)
    return str(value)


def extract_clinical_trial(study, domain):
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})
    description = protocol.get("descriptionModule", {})
    conditions = protocol.get("conditionsModule", {})
    design = protocol.get("designModule", {})
    arms = protocol.get("armsInterventionsModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    contacts = protocol.get("contactsLocationsModule", {})
    outcomes = protocol.get("outcomesModule", {})
    references = protocol.get("referencesModule", {})

    interventions = []
    for item in arms.get("interventions", []):
        interventions.append(
            {
                "type": item.get("type", ""),
                "name": item.get("name", ""),
                "description": item.get("description", ""),
            }
        )

    primary_outcomes = []
    for outcome in outcomes.get("primaryOutcomes", []):
        primary_outcomes.append(
            {
                "measure": outcome.get("measure", ""),
                "description": outcome.get("description", ""),
                "time_frame": outcome.get("timeFrame", ""),
            }
        )

    secondary_outcomes = []
    for outcome in outcomes.get("secondaryOutcomes", []):
        secondary_outcomes.append(
            {
                "measure": outcome.get("measure", ""),
                "description": outcome.get("description", ""),
                "time_frame": outcome.get("timeFrame", ""),
            }
        )

    locations = []
    for location in contacts.get("locations", []):
        locations.append(
            {
                "facility": location.get("facility", ""),
                "city": location.get("city", ""),
                "state": location.get("state", ""),
                "country": location.get("country", ""),
            }
        )

    citations = []
    for ref in references.get("references", []):
        citations.append(
            {"pmid": ref.get("pmid", ""), "citation": ref.get("citation", "")}
        )

    return {
        "source": "clinicaltrials",
        "domain": domain,
        "nct_id": identification.get("nctId", ""),
        "brief_title": identification.get("briefTitle", ""),
        "official_title": identification.get("officialTitle", ""),
        "brief_summary": description.get("briefSummary", ""),
        "detailed_description": description.get("detailedDescription", ""),
        "conditions": _ct_to_list(conditions.get("conditions")),
        "keywords": _ct_to_list(conditions.get("keywords")),
        "study_type": design.get("studyType", ""),
        "phase": _ct_join(design.get("phases")),
        "allocation": _ct_get(design, "designInfo", "allocation"),
        "intervention_model": _ct_get(design, "designInfo", "interventionModel"),
        "masking": _ct_get(design, "designInfo", "maskingInfo", "masking"),
        "primary_purpose": _ct_get(design, "designInfo", "primaryPurpose"),
        "enrollment": _ct_get(design, "enrollmentInfo", "count"),
        "recruitment_status": status.get("overallStatus", ""),
        "start_date": status.get("startDateStruct", {}).get("date", ""),
        "completion_date": status.get("completionDateStruct", {}).get("date", ""),
        "primary_completion_date": status.get("primaryCompletionDateStruct", {}).get(
            "date", ""
        ),
        "sponsor": sponsor.get("leadSponsor", {}).get("name", ""),
        "collaborators": [c.get("name", "") for c in sponsor.get("collaborators", [])],
        "sex": eligibility.get("sex", ""),
        "minimum_age": eligibility.get("minimumAge", ""),
        "maximum_age": eligibility.get("maximumAge", ""),
        "healthy_volunteers": eligibility.get("healthyVolunteers", ""),
        "eligibility_criteria": eligibility.get("eligibilityCriteria", ""),
        "interventions": interventions,
        "primary_outcomes": primary_outcomes,
        "secondary_outcomes": secondary_outcomes,
        "locations": locations,
        "references": citations,
    }


def clinical_trial_to_document(trial: dict) -> str:
    lines = []
    lines.append(f'This document contains information about the clinical study "{trial["brief_title"]}".')
    lines.append(f"The ClinicalTrials.gov identifier is {trial['nct_id']}.")
    if trial["official_title"]:
        lines.append(f"Official title: {trial['official_title']}.")
    if trial["study_type"]:
        lines.append(f"Study type: {trial['study_type']}.")
    if trial["phase"]:
        lines.append(f"Phase: {trial['phase']}.")
    if trial["recruitment_status"]:
        lines.append(f"Recruitment status: {trial['recruitment_status']}.")
    if trial["sponsor"]:
        lines.append(f"Lead sponsor: {trial['sponsor']}.")
    if trial["conditions"]:
        lines.append("Conditions studied: " + ", ".join(trial["conditions"]) + ".")
    if trial["interventions"]:
        lines.append("\nInterventions")
        for i in trial["interventions"]:
            lines.append(f"- {i['type']}: {i['name']}")
            if i["description"]:
                lines.append(i["description"])
    if trial["brief_summary"]:
        lines.append("\nBrief Summary")
        lines.append(trial["brief_summary"])
    if trial["detailed_description"]:
        lines.append("\nDetailed Description")
        lines.append(trial["detailed_description"])
    if trial["eligibility_criteria"]:
        lines.append("\nEligibility")
        lines.append(trial["eligibility_criteria"])
    if trial["primary_outcomes"]:
        lines.append("\nPrimary Outcomes")
        for o in trial["primary_outcomes"]:
            lines.append(f"- {o['measure']}")
            if o["time_frame"]:
                lines.append(f"Time Frame: {o['time_frame']}")
            if o["description"]:
                lines.append(o["description"])
    if trial["secondary_outcomes"]:
        lines.append("\nSecondary Outcomes")
        for o in trial["secondary_outcomes"]:
            lines.append(f"- {o['measure']}")
    if trial["references"]:
        lines.append("\nScientific References")
        for r in trial["references"]:
            citation = r["citation"]
            if r["pmid"]:
                citation += f" (PMID: {r['pmid']}"
            lines.append(citation)
    lines.append("\nThe source of this information is ClinicalTrials.gov.")
    return "\n".join(lines)


def clinical_trial_content_hash(trial: dict) -> str:
    return hashlib.sha256(
        json.dumps(trial, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def pubmed_content_hash(metadata: dict) -> str:
    payload = {
        "title": metadata["title"],
        "abstract": metadata["abstract"],
        "journal": metadata["journal"],
        "doi": metadata["doi"],
        "publication_date": metadata["publication_date"],
        "authors": metadata["authors"],
        "mesh_terms": metadata["mesh_terms"],
        "last_revision": metadata["last_revision"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def wikidata_content_hash(document: dict) -> str:
    payload = {
        "labels": document["labels"],
        "descriptions": document["descriptions"],
        "aliases": document["aliases"],
        "claims": document["claims"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def openalex_content_hash(document: dict) -> str:
    hash_payload = {
        "id": document.get("id"),
        "title": document.get("title"),
        "authors": document.get("authors", []),
        "institutions": document.get("institutions", []),
        "abstract": document.get("abstract", ""),
        "concepts": document.get("concepts", []),
        "referenced_works": document.get("referenced_works", []),
        "doi": document.get("doi"),
        "language": document.get("language"),
        "publication_type": document.get("publication_type", ""),
        "publication_year": document.get("publication_year"),
        "publication_date": document.get("publication_date", ""),
        "journal": document.get("journal", ""),
        "volume": document.get("volume"),
        "issue": document.get("issue"),
        "first_page": document.get("first_page"),
        "last_page": document.get("last_page"),
        "citation_count": document.get("citation_count", 0),
        "landing_url": document.get("landing_url", ""),
        "pdf_url": document.get("pdf_url", ""),
        "open_access": document.get("open_access", {}),
        "updated_date": document.get("updated_date", ""),
        "type": document.get("type", ""),
    }
    canonical = json.dumps(
        hash_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

_redis_client: redis.Redis | None = None

def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client

async def is_ingestion_enabled() -> bool:
    r = get_redis()
    val = await r.get(INGESTION_ENABLED_KEY)
    if val is None:
        return False
    return val == "1"

async def set_ingestion_enabled(enabled: bool):
    r = get_redis()
    await r.set(INGESTION_ENABLED_KEY, "1" if enabled else "0")