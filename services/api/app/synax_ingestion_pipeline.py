import os
import json
import random
import asyncio
import httpx
from xml.etree.ElementTree import ET
from datetime import datetime
from typing import Dict, List

from bs4 import BeautifulSoup, Tag

from services.api.app.synax_config import WIKI_LANG
from services.api.app.synax_research_workspaces import (
    get_db,
    get_ingestion_pipeline_state,
    upsert_ingestion_pipeline_state,
)
from services.api.app.synax_ingestion_helper_functions import (
    MONTH_MAP,
    clean_text,
    sanitize_filename,
    save_text,
    reconstruct_abstract,
    extract_pmc_structure,
    batch_fetch_wikidata_entities,
    batch_resolve_wikidata_labels,
    save_wikidata_cache,
    extract_pdf_text,
    clinical_trial_content_hash,
    pubmed_content_hash,
    wikidata_content_hash,
    openalex_content_hash,
    extract_clinical_trial,
    clinical_trial_to_document,
    BatchCommitter,
)


async def download_wikipedia_articles(
    domains_keywords: Dict[str, List[str]],
    max_articles_per_domain: int = 5000,
    api_batch_size: int = 50,
    max_concurrent_requests: int = 20,
):
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    lang_semaphore = asyncio.Semaphore(3)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0),
        follow_redirects=True,
        http2=True,
        limits=httpx.Limits(
            max_connections=100, max_keepalive_connections=20
        ),
        headers={
            "User-Agent": "SynaX Research Crawler/1.0 (dominiondickson510@gmail.com)"
        },
    ) as client:

        async def wikipedia_get(url, params, retries=4):
            for attempt in range(retries):
                async with semaphore:
                    resp = await client.get(url, params=params)
                if resp.status_code in (429, 503):
                    retry_after = int(resp.headers.get("Retry-After", 0))
                    sleep_for = retry_after or 2**attempt
                    if "maxlag" in resp.text.lower():
                        sleep_for = max(sleep_for, 5)
                    print(
                        f"[Wikipedia Throttle] {resp.status_code} sleep {sleep_for}s"
                    )
                    await asyncio.sleep(sleep_for)
                    continue
                resp.raise_for_status()
                return resp.json()
            print(f"[Wikipedia Give Up] {url} {params}")
            return {}

        async def fetch_page_metadata(lang: str, page_ids: List[str]):
            pages = {}

            async def fetch_batch(batch: List[str]):
                data = await wikipedia_get(
                    f"https://{lang}.wikipedia.org/w/api.php",
                    {
                        "action": "query",
                        "format": "json",
                        "pageids": "|".join(batch),
                        "prop": "info",
                        "inprop": "url",
                    },
                )
                return data.get("query", {}).get("pages", {})

            batches = [
                page_ids[i : i + api_batch_size]
                for i in range(0, len(page_ids), api_batch_size)
            ]
            for result in await asyncio.gather(
                *(fetch_batch(batch) for batch in batches)
            ):
                pages.update(result)
            return pages

        async def fetch_page_structure(lang: str, page_id: str):
            data = await wikipedia_get(
                f"https://{lang}.wikipedia.org/w/api.php",
                {
                    "action": "parse",
                    "format": "json",
                    "pageid": page_id,
                    "prop": "text|sections|images",
                    "disabletoc": 1,
                },
            )
            return data.get("parse")

        async def search_keyword(lang: str, keyword: str):
            try:
                data = await wikipedia_get(
                    f"https://{lang}.wikipedia.org/w/api.php",
                    {
                        "action": "query",
                        "format": "json",
                        "list": "search",
                        "srsearch": keyword,
                        "srlimit": 50,
                    },
                )
                return keyword, data.get("query", {}).get("search", [])
            except Exception as e:
                print(f"[Wikipedia Search Error] {lang}:{keyword}: {e}")
                return keyword, []

        def text_of(node):
            return clean_text(node.get_text(" ", strip=True))

        def table_of(table):
            rows = []
            for tr in table.find_all("tr", recursive=False):
                cells = [
                    text_of(cell)
                    for cell in tr.find_all(["th", "td"], recursive=False)
                ]
                if any(cells):
                    rows.append(cells)
            return rows

        def figure_of(figure):
            image = figure.find("img")
            caption = figure.find("figcaption")
            return {
                "type": "figure",
                "image_url": (
                    (image.get("src") or image.get("data-src") or "")
                    if image
                    else ""
                ),
                "caption": text_of(caption) if caption else "",
            }

        def list_of(element):
            return {
                "type": "list",
                "ordered": element.name == "ol",
                "items": [
                    x
                    for x in (
                        text_of(li)
                        for li in element.find_all("li", recursive=False)
                    )
                    if x
                ],
            }

        def definition_list_of(element):
            return {
                "type": "definition_list",
                "items": [
                    {"type": child.name, "text": value}
                    for child in element.find_all(["dt", "dd"], recursive=False)
                    if (value := text_of(child))
                ],
            }

        def parse_wikipedia_structure(html: str):
            soup = BeautifulSoup(html, "html.parser")
            for node in soup.select("script,style,noscript"):
                node.decompose()

            root = {"type": "document", "content": []}
            stack = []

            def container():
                return stack[-1]["content"] if stack else root["content"]

            def walk(parent):
                for element in parent.children:
                    if not isinstance(element, Tag):
                        continue
                    name = element.name
                    if name in {"h2", "h3", "h4", "h5", "h6"}:
                        title = text_of(element)
                        if not title:
                            continue
                        level = int(name[1])
                        while stack and stack[-1]["level"] >= level:
                            stack.pop()
                        section = {
                            "type": "section",
                            "title": title,
                            "level": level,
                            "content": [],
                        }
                        (
                            stack[-1]["content"]
                            if stack
                            else root["content"]
                        ).append(section)
                        stack.append(section)
                        continue
                    if name == "p":
                        value = text_of(element)
                        if value:
                            container().append({"type": "paragraph", "text": value})
                        continue
                    if name == "table":
                        rows = table_of(element)
                        if rows:
                            container().append({"type": "table", "rows": rows})
                        continue
                    if name == "figure":
                        container().append(figure_of(element))
                        continue
                    if name in {"ul", "ol"}:
                        value = list_of(element)
                        if value["items"]:
                            container().append(value)
                        continue
                    if name == "dl":
                        value = definition_list_of(element)
                        if value["items"]:
                            container().append(value)
                        continue
                    if name in {
                        "div",
                        "section",
                        "article",
                        "main",
                        "blockquote",
                        "center",
                        "dd",
                        "dt",
                    }:
                        walk(element)

            walk(soup)
            return root

        def flatten_structure(structure: dict):
            parts = []

            def walk(nodes):
                for node in nodes:
                    kind = node.get("type")
                    if kind == "section":
                        title = node.get("title", "")
                        if title:
                            parts.append(title)
                        walk(node.get("content", []))
                    elif kind == "paragraph":
                        if node.get("text"):
                            parts.append(node["text"])
                    elif kind == "table":
                        parts.extend(
                            " | ".join(row)
                            for row in node.get("rows", [])
                            if row
                        )
                    elif kind == "figure":
                        if node.get("caption"):
                            parts.append(f"Figure: {node['caption']}")
                    elif kind == "list":
                        parts.extend(node.get("items", []))
                    elif kind == "definition_list":
                        parts.extend(
                            x["text"]
                            for x in node.get("items", [])
                            if x.get("text")
                        )

            walk(structure.get("content", []))
            return clean_text("\n\n".join(x for x in parts if x))

        async def process_page(
            domain, lang, external_id, page, parse_data, existing_state
        ):
            if (
                not page
                or "missing" in page
                or page.get("invalid")
                or not parse_data
            ):
                return False, None

            page_id = str(page.get("pageid"))
            title = page.get("title") or f"id_{page_id}"
            rev_id = str(page.get("lastrevid", ""))
            safe_title = sanitize_filename(title) or f"untitled_{page_id}"
            filename = (
                existing_state["filename"]
                if existing_state and existing_state.get("filename")
                else f"{domain}_{lang}_{page_id}_{safe_title}.txt"
            )
            html = parse_data.get("text", {}).get("*", "")
            if not html:
                return False, None

            structure = parse_wikipedia_structure(html)
            text = flatten_structure(structure)

            if len(text.split()) < 500:
                return False, None

            metadata = {
                "source": "wikipedia",
                "domain": domain,
                "language": lang,
                "page_id": page_id,
                "title": title,
                "revid": rev_id,
                "url": page.get(
                    "fullurl",
                    f"https://{lang}.wikipedia.org/?curid={page_id}",
                ),
                "content_model": "structured",
                "content_types": [
                    "section",
                    "paragraph",
                    "table",
                    "figure",
                    "list",
                    "definition_list",
                ],
                "images": parse_data.get("images", []),
                "sections": parse_data.get("sections", []),
                "structure": structure,
            }

            await asyncio.to_thread(save_text, text, domain, filename)
            await asyncio.to_thread(
                save_text,
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                domain,
                os.path.splitext(filename)[0] + ".json",
            )

            return True, {
                "source": "wikipedia",
                "external_id": external_id,
                "filename": filename,
                "meta": {"revid": rev_id},
            }

        async def download_lang(lang: str):
            async with lang_semaphore:
                for domain, keywords in domains_keywords.items():
                    count = 0
                    discovered_ids: set[str] = set()
                    existing_states: Dict[str, dict | None] = {}

                    async with get_db() as lookup_session:
                        search_results = await asyncio.gather(
                            *(search_keyword(lang, kw) for kw in keywords)
                        )
                        for _, results in search_results:
                            for result in results:
                                page_id = str(result["pageid"])
                                if page_id not in discovered_ids:
                                    discovered_ids.add(page_id)

                        for page_id in discovered_ids:
                            external_id = f"{lang}:{page_id}"
                            existing = await get_ingestion_pipeline_state(
                                session=lookup_session,
                                source="wikipedia",
                                external_id=external_id,
                            )
                            existing_states[page_id] = (
                                {
                                    "filename": existing.filename,
                                    "revid": (existing.meta or {}).get("revid"),
                                }
                                if existing
                                else None
                            )

                    if not discovered_ids:
                        continue

                    pages = await fetch_page_metadata(lang, list(discovered_ids))
                    page_ids_to_fetch = []

                    for page_id in discovered_ids:
                        page = pages.get(page_id)
                        if not page:
                            continue
                        existing = existing_states.get(page_id)
                        revid = str(page.get("lastrevid", ""))
                        if not (
                            existing
                            and existing["revid"] == revid
                            and revid != ""
                        ):
                            page_ids_to_fetch.append(page_id)

                    page_ids_to_fetch = page_ids_to_fetch[:max_articles_per_domain]

                    async def fetch_structure(page_id):
                        try:
                            return page_id, await fetch_page_structure(
                                lang, page_id
                            )
                        except Exception as e:
                            print(
                                f"[Wikipedia Parse Error] {lang}:{page_id}: {e}"
                            )
                            return page_id, None

                    structure_results = await asyncio.gather(
                        *(
                            fetch_structure(page_id)
                            for page_id in page_ids_to_fetch
                        )
                    )

                    async with get_db() as write_session:
                        committer = BatchCommitter(write_session, batch_size=50)
                        try:
                            for page_id, parse_data in structure_results:
                                if count >= max_articles_per_domain:
                                    break
                                if not parse_data:
                                    continue
                                page = pages.get(page_id)
                                if not page:
                                    continue
                                existing = existing_states.get(page_id)
                                revid = str(page.get("lastrevid", ""))
                                if (
                                    existing
                                    and existing["revid"] == revid
                                    and revid != ""
                                ):
                                    continue
                                success, state_row = await process_page(
                                    domain,
                                    lang,
                                    f"{lang}:{page_id}",
                                    page,
                                    parse_data,
                                    existing,
                                )
                                if success and state_row:
                                    await upsert_ingestion_pipeline_state(
                                        session=write_session, **state_row
                                    )
                                    await committer.flush()
                                    count += 1
                            await committer.finish()
                        except Exception:
                            await committer.rollback()
                            raise

                    print(f"[Wikipedia] {lang} | {domain}: {count}")

        results = await asyncio.gather(
            *(download_lang(lang) for lang in WIKI_LANG),
            return_exceptions=True,
        )
        for lang, res in zip(WIKI_LANG, results):
            if isinstance(res, Exception):
                print(f"[Wikipedia Fatal] {lang} crashed: {res}")


async def download_arxiv_papers(
    domain, query, max_results=10000
):
    base_url = "https://export.arxiv.org/api/query"
    page_size = 500
    semaphore = asyncio.Semaphore(4)
    max_retries = 3

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0),
        follow_redirects=True,
        http2=True,
        limits=httpx.Limits(
            max_connections=50, max_keepalive_connections=20
        ),
        headers={
            "User-Agent": "SynaX Research Crawler/1.0 (dominiondickson510@gmail.com)"
        },
    ) as client:

        async def fetch_page(start):
            params = {
                "search_query": query,
                "start": start,
                "max_results": min(page_size, max_results - start),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            for attempt in range(1, max_retries + 1):
                async with semaphore:
                    try:
                        response = await client.get(base_url, params=params)
                        response.raise_for_status()
                        return response.text
                    except Exception as e:
                        if attempt == max_retries:
                            print(
                                f"[arXiv Fetch Error] {query} (start={start}) after {max_retries} attempts: {e}"
                            )
                        else:
                            print(
                                f"[arXiv Retry] {query} (start={start}) attempt={attempt}/{max_retries}: {e}"
                            )
                            await asyncio.sleep(2 ** (attempt - 1))
            return None

        count = 0
        async with get_db() as db_session:
            committer = BatchCommitter(db_session, batch_size=50)
            try:
                for start in range(0, max_results, page_size):
                    xml = await fetch_page(start)
                    if not xml:
                        print(
                            f"[arXiv Page Skipped] {query} (start={start})"
                        )
                        continue
                    try:
                        root = ET.fromstring(xml)
                    except ET.ParseError as e:
                        print(
                            f"[arXiv XML Parse Error] {query} (start={start}): {e}"
                        )
                        continue

                    ns = {
                        "atom": "http://www.w3.org/2005/Atom",
                        "arxiv": "http://arxiv.org/schemas/atom",
                    }
                    entries = root.findall("atom:entry", ns)
                    if not entries:
                        break

                    for entry in entries:
                        entry_id = ""
                        try:
                            entry_id = entry.findtext(
                                "atom:id", default="", namespaces=ns
                            ).strip()
                            if not entry_id:
                                continue

                            title = entry.findtext(
                                "atom:title", default="", namespaces=ns
                            ).strip()
                            summary = entry.findtext(
                                "atom:summary", default="", namespaces=ns
                            ).strip()

                            authors = []
                            for author in entry.findall("atom:author", ns):
                                name = author.findtext(
                                    "atom:name", default="", namespaces=ns
                                ).strip()
                                if name:
                                    authors.append(name)

                            published = entry.findtext(
                                "atom:published", default="", namespaces=ns
                            ).strip()
                            updated = entry.findtext(
                                "atom:updated", default="", namespaces=ns
                            ).strip()

                            primary = entry.find("arxiv:primary_category", ns)
                            primary_category = (
                                primary.attrib.get("term", "")
                                if primary is not None
                                else ""
                            )

                            categories = [
                                category.attrib.get("term")
                                for category in entry.findall(
                                    "atom:category", ns
                                )
                                if category.attrib.get("term")
                            ]

                            doi_elem = entry.find("arxiv:doi", ns)
                            doi = (
                                doi_elem.text.strip()
                                if doi_elem is not None and doi_elem.text
                                else ""
                            )

                            journal_ref_elem = entry.find(
                                "arxiv:journal_ref", ns
                            )
                            journal_ref = (
                                journal_ref_elem.text.strip()
                                if journal_ref_elem is not None
                                and journal_ref_elem.text
                                else ""
                            )

                            pdf_url = ""
                            for link in entry.findall("atom:link", ns):
                                if link.attrib.get("title") == "pdf":
                                    pdf_url = link.attrib.get("href", "")
                                    break

                            publication_status = (
                                "journal" if journal_ref else "preprint"
                            )
                            peer_reviewed = bool(journal_ref)

                            existing = (
                                await get_ingestion_pipeline_state(
                                    session=db_session,
                                    source="arxiv",
                                    external_id=entry_id,
                                )
                            )

                            if existing:
                                existing_meta = existing.meta or {}
                                old_updated = existing_meta.get("updated", "")
                                old_journal_ref = existing_meta.get(
                                    "journal_ref", ""
                                )
                                old_publication_status = existing_meta.get(
                                    "publication_status", "preprint"
                                )
                                old_peer_reviewed = existing_meta.get(
                                    "peer_reviewed", False
                                )
                                metadata_changed = (
                                    old_updated != updated
                                    or old_journal_ref != journal_ref
                                    or old_publication_status
                                    != publication_status
                                    or old_peer_reviewed != peer_reviewed
                                )
                                if not metadata_changed:
                                    continue

                            full_text = ""
                            if pdf_url:
                                full_text = await extract_pdf_text(
                                    client, pdf_url, semaphore
                                )

                            metadata = {
                                "source": "arxiv",
                                "domain": domain,
                                "query": query,
                                "entry_id": entry_id,
                                "title": title,
                                "authors": authors,
                                "abstract": summary,
                                "full_text": full_text,
                                "published": published,
                                "updated": updated,
                                "primary_category": primary_category,
                                "categories": categories,
                                "doi": doi,
                                "journal_ref": journal_ref,
                                "publication_status": publication_status,
                                "peer_reviewed": peer_reviewed,
                                "pdf_url": pdf_url,
                            }
                            filename = f"arxiv_{sanitize_filename(entry_id.rsplit('/', 1)[-1])}.json"

                            await asyncio.to_thread(
                                save_text,
                                json.dumps(
                                    metadata,
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                domain,
                                filename,
                            )
                            await upsert_ingestion_pipeline_state(
                                session=db_session,
                                source="arxiv",
                                external_id=entry_id,
                                filename=filename,
                                meta={
                                    "updated": updated,
                                    "journal_ref": journal_ref,
                                    "publication_status": publication_status,
                                    "peer_reviewed": peer_reviewed,
                                },
                            )
                            await committer.flush()
                            count += 1
                        except Exception as e:
                            print(
                                f"[arXiv Paper Error] {entry_id or 'unknown'}: {e}"
                            )

                await committer.finish()
            except Exception:
                await committer.rollback()
                raise

    print(f"[arXiv] {domain}: {count}")


async def download_clinicaltrials(
    domains_queries, page_size=1000, max_pages=1000
):
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    http_semaphore = asyncio.Semaphore(8)
    query_semaphore = asyncio.Semaphore(5)
    total_downloaded = 0

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0),
        follow_redirects=True,
        http2=True,
        limits=httpx.Limits(
            max_connections=50, max_keepalive_connections=20
        ),
        headers={
            "User-Agent": "SynaX Research Crawler/1.0 (dominiondickson510@gmail.com)"
        },
    ) as client:

        async def http_get(url, params=None, retries=6, initial_delay=1.0):
            delay = initial_delay
            for attempt in range(retries):
                try:
                    async with http_semaphore:
                        response = await client.get(url, params=params)
                    if response.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError(
                            f"Retryable status {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    return response
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (400, 401, 403, 404):
                        raise
                    if attempt == retries - 1:
                        raise
                except (
                    httpx.TimeoutException,
                    httpx.NetworkError,
                ):
                    if attempt == retries - 1:
                        raise
                await asyncio.sleep(
                    delay + random.uniform(0, delay * 0.25)
                )
                delay *= 2

        async def process_query(domain, query):
            async with query_semaphore:
                local_count = 0
                async with get_db() as db_session:
                    committer = BatchCommitter(db_session, batch_size=50)
                    try:
                        page_token = None
                        prev_page_token = None
                        pages_processed = 0

                        while pages_processed < max_pages:
                            if (
                                page_token is not None
                                and page_token == prev_page_token
                            ):
                                print(
                                    f"[ClinicalTrials] Duplicate pageToken detected for '{query}', breaking."
                                )
                                break

                            prev_page_token = page_token
                            params = {
                                "query.term": query,
                                "pageSize": page_size,
                                "format": "json",
                            }
                            if page_token:
                                params["pageToken"] = page_token

                            response = await http_get(base_url, params)
                            payload = response.json()
                            studies = payload.get("studies", [])

                            if not studies:
                                break

                            for study in studies:
                                nct_id = "unknown"
                                try:
                                    protocol = study.get(
                                        "protocolSection", {}
                                    )
                                    ident = protocol.get(
                                        "identificationModule", {}
                                    )
                                    status = protocol.get("statusModule", {})
                                    nct_id = (
                                        ident.get("nctId", "").strip()
                                    )

                                    if not nct_id:
                                        continue

                                    last_update = status.get(
                                        "lastUpdatePostDateStruct", {}
                                    ).get("date", "")

                                    state = (
                                        await get_ingestion_pipeline_state(
                                            session=db_session,
                                            source="clinicaltrials",
                                            external_id=nct_id,
                                        )
                                    )

                                    trial = extract_clinical_trial(
                                        study, domain
                                    )
                                    trial_hash = clinical_trial_content_hash(
                                        trial
                                    )

                                    if state:
                                        meta = state.meta or {}
                                        if (
                                            meta.get("last_update")
                                            == last_update
                                            and meta.get("content_hash")
                                            == trial_hash
                                        ):
                                            continue

                                    filename = f"clinicaltrial_{sanitize_filename(nct_id)}.json"
                                    text_filename = f"clinicaltrial_{sanitize_filename(nct_id)}.txt"

                                    await asyncio.to_thread(
                                        save_text,
                                        json.dumps(
                                            trial,
                                            ensure_ascii=False,
                                            indent=2,
                                        ),
                                        domain,
                                        filename,
                                    )
                                    await asyncio.to_thread(
                                        save_text,
                                        clinical_trial_to_document(trial),
                                        domain,
                                        text_filename,
                                    )

                                    await upsert_ingestion_pipeline_state(
                                        session=db_session,
                                        source="clinicaltrials",
                                        external_id=nct_id,
                                        filename=filename,
                                        meta={
                                            "downloaded": True,
                                            "last_update": last_update,
                                            "content_hash": trial_hash,
                                        },
                                    )
                                    await committer.flush()
                                    local_count += 1
                                except Exception as e:
                                    print(
                                        f"[ClinicalTrials Study Error] {nct_id}: {e}"
                                    )

                            pages_processed += 1
                            page_token = payload.get("nextPageToken")

                            if not page_token:
                                break

                            await asyncio.sleep(0.5)

                        await committer.finish()
                        return local_count
                    except Exception:
                        await committer.rollback()
                        raise

        for domain, queries in domains_queries.items():
            results = await asyncio.gather(
                *(process_query(domain, q) for q in queries),
                return_exceptions=True,
            )
            success_count = 0
            for r in results:
                if isinstance(r, Exception):
                    print(
                        f"[ClinicalTrials] {domain}: query failed: {r}"
                    )
                else:
                    success_count += r
            print(f"[ClinicalTrials] {domain}: {success_count}")
            total_downloaded += success_count

    print(f"[ClinicalTrials Total]: {total_downloaded}")


async def download_pubmed_central_articles(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    pmid: str,
    pmcid: str,
    domain: str,
    db_session,
    api_key: str | None = None,
):
    fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    async def http_get(url, params=None, retries=6):
        params = dict(params or {})
        params.update(
            {"tool": "SynaX", "email": "dominiondickson510@gmail.com"}
        )
        if api_key:
            params["api_key"] = api_key

        delay = 1.0
        for attempt in range(retries):
            try:
                async with semaphore:
                    response = await client.get(url, params=params)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"Retryable {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if (
                    e.response is not None
                    and e.response.status_code in (400, 401, 403, 404)
                ):
                    raise
                if attempt == retries - 1:
                    raise
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
            ):
                if attempt == retries - 1:
                    raise
            await asyncio.sleep(
                delay + random.uniform(0, delay * 0.25)
            )
            delay = min(delay * 2, 30)

    try:
        existing = await get_ingestion_pipeline_state(
            session=db_session, source="pubmed", external_id=pmid
        )
        if (
            existing
            and existing.meta.get("pmc_downloaded")
            and existing.meta.get("pmcid") == pmcid
        ):
            return

        response = await http_get(
            fetch, {"db": "pmc", "id": pmcid, "retmode": "xml"}
        )
        xml_text = response.text.strip()

        if not xml_text:
            return

        structured_full_text = extract_pmc_structure(
            xml_text, pmid, pmcid, domain
        )

        if not structured_full_text:
            return

        filename = f"pmc_{pmcid}.json"
        await asyncio.to_thread(
            save_text,
            json.dumps(
                structured_full_text, ensure_ascii=False, indent=2
            ),
            domain,
            filename,
        )

        metadata = dict(existing.meta or {}) if existing else {}
        metadata.update(
            {
                "pmcid": pmcid,
                "pmc_downloaded": True,
                "has_full_text": True,
                "full_text_source": "pubmedcentral",
                "full_text_filename": filename,
            }
        )

        await upsert_ingestion_pipeline_state(
            session=db_session,
            source="pubmed",
            external_id=pmid,
            filename=existing.filename
            if existing
            else f"pubmed_{pmid}.json",
            meta=metadata,
        )
    except Exception as e:
        print(f"[PMC Error] PMID={pmid} PMCID={pmcid}: {e}")


async def download_pubmed_articles(
    domains_queries: Dict[str, List[str]],
    max_results: int = 10000,
    batch_size: int = 500,
    api_key: str | None = None,
    max_concurrent_queries: int = 5,
):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    total_downloaded = 0
    http_sem = asyncio.Semaphore(10)
    query_sem = asyncio.Semaphore(max_concurrent_queries)
    limits = httpx.Limits(
        max_connections=50, max_keepalive_connections=20
    )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=30.0),
        follow_redirects=True,
        http2=False,
        limits=limits,
        headers={
            "User-Agent": "SynaX Research Crawler/1.0 (dominiondickson510@gmail.com)"
        },
    ) as client:

        async def http_get(url, params=None, retries=6):
            params = dict(params or {})
            params.update(
                {"tool": "SynaX", "email": "dominiondickson510@gmail.com"}
            )
            if api_key:
                params["api_key"] = api_key

            delay = 1.0
            for attempt in range(retries):
                try:
                    async with http_sem:
                        r = await client.get(url, params=params)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError(
                            f"Retryable {r.status_code}",
                            request=r.request,
                            response=r,
                        )
                    r.raise_for_status()
                    return r
                except httpx.HTTPStatusError as e:
                    if (
                        e.response is not None
                        and e.response.status_code in (400, 401, 403, 404)
                    ):
                        raise
                    if attempt == retries - 1:
                        raise
                except (
                    httpx.TimeoutException,
                    httpx.NetworkError,
                ):
                    if attempt == retries - 1:
                        raise
                await asyncio.sleep(
                    delay + random.uniform(0, delay * 0.25)
                )
                delay = min(delay * 2, 30)

        async def upsert_article(
            db_session, committer, domain, query, article
        ):
            pmid = article["pmid"]
            existing = await get_ingestion_pipeline_state(
                session=db_session,
                source="pubmed",
                external_id=pmid,
            )

            base_meta = {
                "pmid": pmid,
                "pmcid": article.get("pmcid", ""),
                "source": "pubmed",
                "domain": domain,
                "query": query,
                "title": article["title"],
                "abstract": article["abstract"],
                "journal": article["journal"],
                "doi": article["doi"],
                "publication_date": article["publication_date"],
                "authors": article["authors"],
                "mesh_terms": article["mesh_terms"],
                "last_revision": article["last_revision"],
                "has_full_text": (
                    existing.meta.get("has_full_text", False)
                    if existing
                    else False
                ),
                "full_text_source": (
                    existing.meta.get("full_text_source")
                    if existing
                    else None
                ),
            }

            content_hash = pubmed_content_hash(base_meta)

            if (
                existing
                and existing.meta.get("content_hash") == content_hash
            ):
                return False

            base_meta["content_hash"] = content_hash
            filename = (
                existing.filename
                if existing and existing.meta.get("domain") == domain
                else f"pubmed_{pmid}.json"
            )

            await asyncio.to_thread(
                save_text,
                json.dumps(base_meta, ensure_ascii=False, indent=2),
                domain,
                filename,
            )
            await upsert_ingestion_pipeline_state(
                session=db_session,
                source="pubmed",
                external_id=pmid,
                filename=filename,
                meta=base_meta,
            )
            await committer.flush()
            return True

        async def process_batch(db_session, committer, domain, query, id_batch):
            try:
                resp = await http_get(
                    fetch,
                    {
                        "db": "pubmed",
                        "id": ",".join(id_batch),
                        "retmode": "xml",
                    },
                )
            except Exception as e:
                print(f"[Fetch Error] {query} {id_batch[0]}..: {e}")
                return 0

            soup = BeautifulSoup(resp.text, "xml")
            count = 0
            pmc_tasks = []

            for art in soup.find_all("PubmedArticle"):
                try:
                    pmid_tag = art.find("PMID")
                    if not pmid_tag:
                        continue
                    pmid = pmid_tag.get_text(strip=True)

                    def _text(tag):
                        return (
                            tag.get_text(" ", strip=True)
                            if tag
                            else ""
                        )

                    pub_date = ""
                    pub = art.find("PubDate")
                    if pub:
                        y = _text(pub.find("Year"))
                        m = _text(pub.find("Month"))
                        d = _text(pub.find("Day"))
                        m = MONTH_MAP.get(
                            m, m.zfill(2) if m.isdigit() else "01"
                        )
                        if y:
                            pub_date = (
                                f"{y}-{m or '01'}-{(d or '01').zfill(2)}"
                            )

                    last_rev = ""
                    hist = art.find("History")
                    if hist:
                        dates = []
                        for pr in hist.find_all("PubMedPubDate"):
                            yy = _text(pr.find("Year"))
                            mm = _text(pr.find("Month"))
                            dd = _text(pr.find("Day"))
                            try:
                                mm = MONTH_MAP.get(mm, mm)
                                dates.append(
                                    datetime(
                                        int(yy), int(mm), int(dd or 1)
                                    )
                                )
                            except Exception:
                                continue
                        if dates:
                            last_rev = max(dates).strftime(
                                "%Y-%m-%d"
                            )

                    title = _text(art.find("ArticleTitle"))
                    abstracts = art.find_all("AbstractText")
                    abstract = "\n".join(
                        f"{a.get('Label', '')}: {a.get_text(' ', strip=True)}".strip(
                            ": "
                        )
                        for a in abstracts
                    )

                    journal_tag = art.find("Journal")
                    journal = (
                        _text(journal_tag.find("Title"))
                        if journal_tag
                        else ""
                    )

                    doi = _text(
                        art.find("ArticleId", attrs={"IdType": "doi"})
                    )
                    pmc = _text(
                        art.find("ArticleId", attrs={"IdType": "pmc"})
                    )

                    authors = []
                    for au in art.find_all("Author"):
                        if au.find("CollectiveName"):
                            authors.append(
                                _text(au.find("CollectiveName"))
                            )
                            continue
                        fn = _text(au.find("ForeName"))
                        ln = _text(au.find("LastName"))
                        authors.append(
                            " ".join(p for p in (fn, ln) if p)
                        )

                    mesh = [
                        m.get_text(strip=True)
                        for m in art.find_all("DescriptorName")
                    ]

                    doc = {
                        "pmid": pmid,
                        "pmcid": pmc,
                        "title": title,
                        "abstract": abstract,
                        "journal": journal,
                        "doi": doi,
                        "publication_date": pub_date,
                        "authors": authors,
                        "mesh_terms": mesh,
                        "last_revision": last_rev,
                    }

                    await upsert_article(
                        db_session, committer, domain, query, doc
                    )

                    if pmc:
                        pmc_tasks.append(
                            asyncio.create_task(
                                download_pubmed_central_articles(
                                    client=client,
                                    semaphore=http_sem,
                                    pmid=pmid,
                                    pmcid=pmc,
                                    domain=domain,
                                    db_session=db_session,
                                    api_key=api_key,
                                )
                            )
                        )
                    count += 1
                except Exception as e:
                    print(f"[Article Error] {e}")

            if pmc_tasks:
                await asyncio.gather(
                    *pmc_tasks, return_exceptions=True
                )

            return count

        async def process_query(domain, query):
            async with query_sem:
                try:
                    r = await http_get(
                        base,
                        {
                            "db": "pubmed",
                            "term": query,
                            "retmax": max_results,
                            "retmode": "json",
                        },
                    )
                    data = r.json().get("esearchresult", {})
                except Exception as e:
                    print(f"[Search Error] {query}: {e}")
                    return 0

                if "errorlist" in data:
                    print(
                        f"[Search Warning] {query}: {data['errorlist']}"
                    )

                ids = data.get("idlist", [])
                if not ids:
                    return 0

                async with get_db() as db_session:
                    committer = BatchCommitter(db_session, batch_size=50)
                    try:
                        total = 0
                        for i in range(0, len(ids), batch_size):
                            total += await process_batch(
                                db_session,
                                committer,
                                domain,
                                query,
                                ids[i : i + batch_size],
                            )
                        await committer.finish()
                        return total
                    except Exception:
                        await committer.rollback()
                        raise

        for domain, queries in domains_queries.items():
            results = await asyncio.gather(
                *(process_query(domain, q) for q in queries)
            )
            print(f"[PubMed] {domain}: {sum(results)}")
            total_downloaded += sum(results)

    print(f"[Total] {total_downloaded}")


async def download_wikidata_entities(
    domains_keywords: Dict[str, List[str]],
    max_results: int = 5000,
    search_limit_per_keyword: int = 50,
):
    search_api = "https://www.wikidata.org/w/api.php"
    semaphore = asyncio.Semaphore(10)
    client_limits = httpx.Limits(
        max_connections=50, max_keepalive_connections=20
    )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0),
        follow_redirects=True,
        http2=True,
        limits=client_limits,
        headers={
            "User-Agent": "SynaX Research Crawler/1.0 (dominiondickson510@gmail.com)"
        },
    ) as client:

        async def search_keyword_with_retry(
            keyword: str, retries: int = 3
        ):
            results = []
            continue_offset = 0

            for attempt in range(retries + 1):
                try:
                    async with semaphore:
                        while len(results) < search_limit_per_keyword:
                            resp = await client.get(
                                search_api,
                                params={
                                    "action": "wbsearchentities",
                                    "search": keyword,
                                    "language": "en",
                                    "format": "json",
                                    "limit": min(
                                        50,
                                        search_limit_per_keyword
                                        - len(results),
                                    ),
                                    "continue": continue_offset,
                                },
                            )
                            resp.raise_for_status()
                            data = resp.json()
                            items = data.get("search", [])

                            if not items:
                                break

                            results.extend(items)

                            if "search-continue" in data:
                                continue_offset = (
                                    data["search-continue"]
                                )
                            else:
                                break

                        return results
                except httpx.HTTPStatusError as e:
                    if (
                        e.response.status_code == 429
                        and attempt < retries
                    ):
                        await asyncio.sleep(2**attempt + 1)
                        continue
                    print(
                        f"[Wikidata Search Error] {keyword}: {e}"
                    )
                    return results if results else None
                except Exception as e:
                    if attempt < retries:
                        await asyncio.sleep(2**attempt)
                        continue
                    print(
                        f"[Wikidata Search Error] {keyword}: {e}"
                    )
                    return None

            return results

        async def process_domain(
            domain: str, keywords: List[str]
        ) -> int:
            written_count = 0
            discovered_qids: set[str] = set()

            for keyword in keywords:
                if len(discovered_qids) >= max_results:
                    break
                search_items = await search_keyword_with_retry(keyword)
                if not search_items:
                    continue

                for item in search_items:
                    qid = item.get("id")
                    if not qid or qid in discovered_qids:
                        continue
                    discovered_qids.add(qid)
                    if len(discovered_qids) >= max_results:
                        break

            if not discovered_qids:
                print(f"[Wikidata] {domain}: 0 (no results)")
                return 0

            qid_list = list(discovered_qids)[:max_results]

            async with get_db() as db_session:
                committer = BatchCommitter(db_session, batch_size=50)
                try:
                    for i in range(0, len(qid_list), 50):
                        chunk_qids = qid_list[i : i + 50]
                        entities = await batch_fetch_wikidata_entities(
                            chunk_qids, client
                        )

                        if not entities:
                            continue

                        all_entity_ids: set[str] = set()
                        all_property_ids: set[str] = set()

                        for entity in entities.values():
                            for pid, claims in entity.get(
                                "claims", {}
                            ).items():
                                all_property_ids.add(pid)
                                for claim in claims[:200]:
                                    mainsnak = claim.get("mainsnak", {})
                                    if (
                                        mainsnak.get("datatype")
                                        == "wikibase-item"
                                    ):
                                        val = mainsnak.get(
                                            "datavalue", {}
                                        ).get("value", {})
                                        if (
                                            isinstance(val, dict)
                                            and val.get("id")
                                        ):
                                            all_entity_ids.add(
                                                val["id"]
                                            )

                        entity_labels, property_labels = (
                            await asyncio.gather(
                                batch_resolve_wikidata_labels(
                                    list(all_entity_ids), client
                                ),
                                batch_resolve_wikidata_labels(
                                    list(all_property_ids), client
                                ),
                            )
                        )

                        for qid, entity in entities.items():
                            labels_data = entity.get("labels", {})
                            descriptions_data = entity.get(
                                "descriptions", {}
                            )
                            aliases_data = entity.get("aliases", {})
                            claims = entity.get("claims", {})
                            modified = entity.get("modified", "")

                            labels = {
                                lang: labels_data.get(lang, {}).get(
                                    "value", ""
                                )
                                for lang in WIKI_LANG
                            }
                            descriptions = {
                                lang: descriptions_data.get(lang, {}).get(
                                    "value", ""
                                )
                                for lang in WIKI_LANG
                            }
                            aliases = {
                                lang: [
                                    a["value"]
                                    for a in aliases_data.get(lang, [])
                                ]
                                for lang in WIKI_LANG
                            }

                            resolved_claims = {}
                            entity_claims = {}

                            for pid, values in claims.items():
                                property_name = property_labels.get(
                                    pid, pid
                                )
                                claim_values = []
                                entity_values = []

                                for claim in values[:200]:
                                    try:
                                        mainsnak = claim.get(
                                            "mainsnak", {}
                                        )
                                        datatype = mainsnak.get(
                                            "datatype"
                                        )
                                        value = mainsnak.get(
                                            "datavalue", {}
                                        ).get("value")

                                        if (
                                            datatype == "wikibase-item"
                                            and isinstance(value, dict)
                                        ):
                                            vid = value.get("id")
                                            label = entity_labels.get(
                                                vid, vid
                                            )
                                            claim_values.append(label)
                                            entity_values.append(label)
                                        elif isinstance(value, str):
                                            claim_values.append(value)
                                        elif isinstance(value, dict):
                                            claim_values.append(
                                                json.dumps(
                                                    value,
                                                    ensure_ascii=False,
                                                )
                                            )
                                        elif value is not None:
                                            claim_values.append(
                                                str(value)
                                            )
                                    except Exception:
                                        continue

                                if claim_values:
                                    resolved_claims[
                                        property_name
                                    ] = claim_values
                                if entity_values:
                                    entity_claims[
                                        property_name
                                    ] = entity_values

                            json_document = {
                                "source": "wikidata",
                                "domain": domain,
                                "qid": qid,
                                "modified": modified,
                                "labels": labels,
                                "descriptions": descriptions,
                                "aliases": aliases,
                                "claims": resolved_claims,
                                "entity_claims": entity_claims,
                            }
                            content_hash = wikidata_content_hash(
                                json_document
                            )
                            json_document["content_hash"] = content_hash

                            existing = await get_ingestion_pipeline_state(
                                session=db_session,
                                source="wikidata",
                                external_id=qid,
                            )

                            if existing:
                                old_modified = existing.meta.get(
                                    "modified", ""
                                )
                                old_hash = existing.meta.get(
                                    "content_hash", ""
                                )
                                if (
                                    old_modified == modified
                                    and old_hash == content_hash
                                ):
                                    continue

                            json_filename = (
                                existing.filename
                                if existing
                                else f"wikidata_{qid}.json"
                            )
                            text_filename = (
                                existing.meta.get("text_document")
                                if existing
                                else f"wikidata_{qid}.txt"
                            )

                            await asyncio.to_thread(
                                save_text,
                                json.dumps(
                                    json_document,
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                domain,
                                json_filename,
                            )

                            entity_name = labels.get("en") or next(
                                (v for v in labels.values() if v),
                                "Unknown Entity",
                            )
                            description = descriptions.get("en") or next(
                                (v for v in descriptions.values() if v),
                                "",
                            )

                            document = (
                                f'This document contains information about the Wikidata entity "{entity_name}".'
                            )
                            document += (
                                f"\n\nThe Wikidata identifier assigned to this entity is {qid}."
                            )

                            if description:
                                document += (
                                    f"\n\nAccording to Wikidata, this entity is described as: {description}"
                                )

                            aliases_en = aliases.get("en", [])
                            if aliases_en:
                                if len(aliases_en) == 1:
                                    document += (
                                        f"\n\nThe entity is also known by the alias {aliases_en[0]}."
                                    )
                                else:
                                    document += (
                                        "\n\nThe entity is also known by the following aliases: "
                                    )
                                    document += (
                                        ", ".join(aliases_en[:-1])
                                        + " and "
                                        + aliases_en[-1]
                                        + "."
                                    )

                            if resolved_claims:
                                document += (
                                    "\n\nThe following structured knowledge is associated with this entity."
                                )
                                for property_name, values in sorted(
                                    resolved_claims.items()
                                ):
                                    if not values:
                                        continue
                                    if len(values) == 1:
                                        document += (
                                            f"\n\n{property_name}: {values[0]}."
                                        )
                                    else:
                                        document += (
                                            f"\n\n{property_name}: "
                                        )
                                        document += (
                                            ", ".join(values[:-1])
                                            + " and "
                                            + values[-1]
                                            + "."
                                        )

                            document += (
                                "\n\nThe source of this information is Wikidata."
                            )

                            await asyncio.to_thread(
                                save_text,
                                clean_text(document),
                                domain,
                                text_filename,
                            )

                            await upsert_ingestion_pipeline_state(
                                session=db_session,
                                source="wikidata",
                                external_id=qid,
                                filename=json_filename,
                                meta={
                                    "downloaded": True,
                                    "modified": modified,
                                    "content_hash": content_hash,
                                    "text_document": text_filename,
                                },
                            )
                            written_count += 1

                        await committer.flush()

                    await committer.finish()
                    print(
                        f"[Wikidata] {domain}: {written_count} written / {len(qid_list)} discovered"
                    )
                    return written_count
                except Exception:
                    await committer.rollback()
                    raise

        domain_counts = await asyncio.gather(
            *(
                process_domain(domain, kws)
                for domain, kws in domains_keywords.items()
            )
        )
        total_count = sum(domain_counts)
        await asyncio.to_thread(save_wikidata_cache)
        print(f"[Wikidata Total] {total_count}")


async def download_openalex(
    domains_queries, max_pages=1000, per_page=200
):
    base_url = "https://api.openalex.org/works"
    semaphore = asyncio.Semaphore(8)
    counter_lock = asyncio.Lock()
    new_count = 0
    updated_count = 0
    unchanged_count = 0

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0),
        follow_redirects=True,
        http2=True,
        limits=httpx.Limits(
            max_connections=50, max_keepalive_connections=20
        ),
        headers={
            "User-Agent": "SynaX Research Crawler/1.0 (dominiondickson510@gmail.com)"
        },
    ) as client:

        async def fetch_with_retry(params, max_retries=5):
            for attempt in range(max_retries + 1):
                response = await client.get(base_url, params=params)
                if response.status_code != 429:
                    return response
                if attempt >= max_retries:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = (
                        float(retry_after)
                        if retry_after
                        else min(2**attempt, 60)
                    )
                except (TypeError, ValueError):
                    delay = min(2**attempt, 60)
                print(
                    f"[OpenAlex 429] Retrying after {delay:g}s (attempt {attempt+1}/{max_retries})"
                )
                await asyncio.sleep(delay)
            return response

        async def process_query(domain, query):
            nonlocal new_count, updated_count, unchanged_count
            local_new = 0
            local_updated = 0
            local_unchanged = 0

            async with get_db() as db_session:
                committer = BatchCommitter(db_session, batch_size=50)
                try:
                    cursor = "*"
                    pages_processed = 0
                    seen_cursors = set()

                    while cursor and pages_processed < max_pages:
                        if cursor in seen_cursors:
                            print(f"[OpenAlex Cursor Loop] {query}")
                            break
                        seen_cursors.add(cursor)

                        async with semaphore:
                            try:
                                response = await fetch_with_retry(
                                    {
                                        "search": query,
                                        "per-page": per_page,
                                        "cursor": cursor,
                                        "mailto": "dominiondickson510@gmail.com",
                                    }
                                )
                                response.raise_for_status()
                                data = response.json()
                            except Exception as e:
                                print(
                                    f"[OpenAlex Fetch Error] {query}: {e}"
                                )
                                break

                        papers = data.get("results", [])
                        if not papers:
                            break

                        for paper in papers:
                            pid = paper.get("id")
                            try:
                                if not pid:
                                    continue

                                authors = []
                                institutions = set()

                                for authorship in paper.get(
                                    "authorships", []
                                ):
                                    author = authorship.get(
                                        "author", {}
                                    ).get("display_name", "")
                                    if author:
                                        authors.append(author)

                                    for inst in authorship.get(
                                        "institutions", []
                                    ):
                                        name = inst.get(
                                            "display_name", ""
                                        )
                                        if name:
                                            institutions.add(name)

                                authors = list(
                                    dict.fromkeys(authors)
                                )
                                institutions = sorted(institutions)

                                try:
                                    abstract = reconstruct_abstract(
                                        paper.get(
                                            "abstract_inverted_index",
                                            {},
                                        )
                                    )
                                except Exception:
                                    abstract = ""

                                title = (
                                    (
                                        paper.get("display_name") or ""
                                    ).strip()
                                )
                                publication_year = paper.get(
                                    "publication_year"
                                )
                                publication_date = paper.get(
                                    "publication_date", ""
                                )
                                citation_count = (
                                    paper.get("cited_by_count") or 0
                                )
                                concepts = [
                                    concept.get(
                                        "display_name", ""
                                    ).strip()
                                    for concept in paper.get(
                                        "concepts", []
                                    )
                                    if concept.get("display_name")
                                ]
                                referenced_works = (
                                    paper.get(
                                        "referenced_works", []
                                    ) or []
                                )
                                doi = paper.get("doi")
                                language = paper.get("language")
                                publication_type = (
                                    (paper.get("type") or "")
                                    .replace("-", " ")
                                    .strip()
                                )

                                open_access = (
                                    paper.get("open_access") or {}
                                )
                                primary_location = (
                                    paper.get("primary_location") or {}
                                )
                                source = (
                                    primary_location.get("source") or {}
                                )
                                journal = (
                                    source.get("display_name") or ""
                                ).strip()
                                landing_url = (
                                    primary_location.get(
                                        "landing_page_url"
                                    ) or ""
                                )
                                pdf_url = (
                                    primary_location.get("pdf_url") or ""
                                )

                                biblio = paper.get("biblio") or {}
                                volume = biblio.get("volume")
                                issue = biblio.get("issue")
                                first_page = biblio.get("first_page")
                                last_page = biblio.get("last_page")
                                updated_date = paper.get(
                                    "updated_date", ""
                                )
                                work_type = paper.get("type", "")

                                current_document = {
                                    "id": pid,
                                    "title": title,
                                    "authors": authors,
                                    "institutions": institutions,
                                    "abstract": abstract,
                                    "concepts": concepts,
                                    "referenced_works": referenced_works,
                                    "doi": doi,
                                    "language": language,
                                    "publication_type": publication_type,
                                    "publication_year": publication_year,
                                    "publication_date": publication_date,
                                    "journal": journal,
                                    "volume": volume,
                                    "issue": issue,
                                    "first_page": first_page,
                                    "last_page": last_page,
                                    "citation_count": citation_count,
                                    "landing_url": landing_url,
                                    "pdf_url": pdf_url,
                                    "open_access": open_access,
                                    "updated_date": updated_date,
                                    "type": work_type,
                                }
                                content_hash = openalex_content_hash(
                                    current_document
                                )

                                existing = (
                                    await get_ingestion_pipeline_state(
                                        session=db_session,
                                        source="openalex",
                                        external_id=pid,
                                    )
                                )

                                if existing:
                                    existing_meta = existing.meta or {}
                                    old_hash = existing_meta.get(
                                        "content_hash", ""
                                    )
                                    if old_hash == content_hash:
                                        local_unchanged += 1
                                        continue
                                    local_updated += 1
                                else:
                                    local_new += 1

                                safe_id = sanitize_filename(
                                    pid.rsplit("/", 1)[-1]
                                )
                                filename = (
                                    existing.filename
                                    if existing and existing.filename
                                    else f"openalex_{safe_id}.txt"
                                )
                                json_filename = (
                                    os.path.splitext(filename)[0]
                                    + ".json"
                                )

                                document = f'This document contains information about the scientific publication "{title}" obtained from the OpenAlex scholarly database.'

                                if publication_type:
                                    article = (
                                        "an"
                                        if publication_type[:1].lower()
                                        in "aeiou"
                                        else "a"
                                    )
                                    document += f"\n\nOpenAlex classifies this work as {article} {publication_type}."

                                if publication_year:
                                    document += (
                                        f"\n\nThe publication year is {publication_year}."
                                    )
                                if publication_date:
                                    document += (
                                        f" The publication date is {publication_date}."
                                    )
                                if journal:
                                    document += f" According to OpenAlex, this work was published in {journal}."

                                if (
                                    volume
                                    or issue
                                    or first_page
                                    or last_page
                                ):
                                    document += " "
                                    if volume:
                                        document += (
                                            f"It appears in volume {volume}"
                                        )
                                        if issue:
                                            document += (
                                                f", issue {issue}"
                                            )
                                    elif issue:
                                        document += (
                                            f"It appears in issue {issue}"
                                        )
                                    if (
                                        first_page and last_page
                                    ):
                                        document += f", on pages {first_page}-{last_page}"
                                    elif first_page:
                                        document += f", beginning on page {first_page}"
                                    document += "."

                                if authors:
                                    if len(authors) == 1:
                                        document += (
                                            f"\n\nThe publication was authored by {authors[0]}."
                                        )
                                    else:
                                        document += (
                                            "\n\nThe publication was authored by "
                                            + ", ".join(
                                                authors[:-1]
                                            )
                                            + " and "
                                            + authors[-1]
                                            + "."
                                        )

                                if institutions:
                                    document += (
                                        "\n\nThe contributing institutions include "
                                        + ", ".join(institutions)
                                        + "."
                                    )

                                document += f"\n\nAt the time this document was downloaded, the publication had received {citation_count:,} citations according to OpenAlex."

                                if concepts:
                                    if len(concepts) == 1:
                                        document += f"\n\nThe primary research concept associated with this publication is {concepts[0]}."
                                    else:
                                        document += (
                                            "\n\nThe publication is associated with the following research concepts: "
                                            + ", ".join(
                                                concepts[:-1]
                                            )
                                            + " and "
                                            + concepts[-1]
                                            + "."
                                        )

                                if abstract:
                                    document += (
                                        f"\n\nAbstract\n\n{abstract}"
                                    )

                                if referenced_works:
                                    document += f"\n\nThis publication references {len(referenced_works)} previous scholarly works."
                                    document += (
                                        "\n\nReferenced OpenAlex works:"
                                    )
                                    for work in referenced_works:
                                        document += f"\n- {work}"

                                if doi:
                                    document += f"\n\nThe Digital Object Identifier (DOI) assigned to this publication is {doi}."

                                if language:
                                    document += (
                                        f"\n\nThe language of the publication is {language}."
                                    )

                                if open_access:
                                    if open_access.get("is_oa"):
                                        document += (
                                            "\n\nAccording to OpenAlex, this publication is available as Open Access."
                                        )
                                        status = open_access.get(
                                            "oa_status"
                                        )
                                        if status:
                                            document += f" Its Open Access status is '{status}'."
                                    else:
                                        document += "\n\nAccording to OpenAlex, this publication is not marked as Open Access."

                                if landing_url:
                                    document += (
                                        f"\n\nThe publication can be accessed through the following landing page:\n{landing_url}"
                                    )

                                if pdf_url:
                                    document += (
                                        f"\n\nA direct PDF is available at:\n{pdf_url}"
                                    )

                                if updated_date:
                                    document += (
                                        f"\n\nThe OpenAlex record was last updated on {updated_date}."
                                    )

                                document += (
                                    f"\n\nThe OpenAlex identifier for this publication is {pid}."
                                )
                                document += (
                                    "\n\nThe source of this information is the OpenAlex scholarly database."
                                )

                                metadata = {
                                    "source": "openalex",
                                    "domain": domain,
                                    "id": pid,
                                    "title": title,
                                    "authors": authors,
                                    "institutions": institutions,
                                    "abstract": abstract,
                                    "concepts": concepts,
                                    "referenced_works": referenced_works,
                                    "doi": doi,
                                    "language": language,
                                    "publication_type": publication_type,
                                    "publication_year": publication_year,
                                    "publication_date": publication_date,
                                    "journal": journal,
                                    "volume": volume,
                                    "issue": issue,
                                    "first_page": first_page,
                                    "last_page": last_page,
                                    "citation_count": citation_count,
                                    "landing_url": landing_url,
                                    "pdf_url": pdf_url,
                                    "open_access": open_access,
                                    "updated_date": updated_date,
                                    "type": work_type,
                                    "content_hash": content_hash,
                                }

                                await asyncio.to_thread(
                                    save_text,
                                    clean_text(document),
                                    domain,
                                    filename,
                                )
                                await asyncio.to_thread(
                                    save_text,
                                    json.dumps(
                                        metadata,
                                        ensure_ascii=False,
                                        indent=2,
                                    ),
                                    domain,
                                    json_filename,
                                )
                                await upsert_ingestion_pipeline_state(
                                    session=db_session,
                                    source="openalex",
                                    external_id=pid,
                                    filename=filename,
                                    meta={
                                        "downloaded": True,
                                        "content_hash": content_hash,
                                        "updated_date": updated_date,
                                        "citation_count": citation_count,
                                        "publication_date": publication_date,
                                        "doi": doi,
                                        "title": title,
                                        "abstract": abstract,
                                        "pdf_url": pdf_url,
                                        "landing_url": landing_url,
                                        "domain": domain,
                                        "id": pid,
                                        "authors": authors,
                                        "institutions": institutions,
                                        "concepts": concepts,
                                        "referenced_works": referenced_works,
                                        "language": language,
                                        "publication_type": publication_type,
                                        "publication_year": publication_year,
                                        "journal": journal,
                                        "volume": volume,
                                        "issue": issue,
                                        "first_page": first_page,
                                        "last_page": last_page,
                                        "open_access": open_access,
                                        "type": work_type,
                                    },
                                )
                                await committer.flush()
                            except Exception as e:
                                print(
                                    f"[OpenAlex Paper Error] {pid or 'unknown'}: {e}"
                                )

                        next_cursor = (
                            data.get("meta", {}).get("next_cursor")
                        )
                        if not next_cursor or next_cursor == cursor:
                            break
                        cursor = next_cursor
                        pages_processed += 1
                        await asyncio.sleep(1)

                    await committer.finish()

                    async with counter_lock:
                        new_count += local_new
                        updated_count += local_updated
                        unchanged_count += local_unchanged

                    return local_new, local_updated, local_unchanged
                except Exception:
                    await committer.rollback()
                    raise

        for domain, queries in domains_queries.items():
            results = await asyncio.gather(
                *(
                    process_query(domain, query) for query in queries
                ),
                return_exceptions=True,
            )
            domain_new = 0
            domain_updated = 0
            domain_unchanged = 0

            for result in results:
                if isinstance(result, Exception):
                    print(
                        f"[OpenAlex] {domain}: query failed: {result}"
                    )
                else:
                    n, u, c = result
                    domain_new += n
                    domain_updated += u
                    domain_unchanged += c

            print(
                f"[OpenAlex] {domain}: new={domain_new};updated={domain_updated};unchanged={domain_unchanged}"
            )