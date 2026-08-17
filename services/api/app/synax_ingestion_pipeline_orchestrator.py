# services/api/app/synax_ingestion_pipeline_orchestrator.py
import asyncio
from services.api.app.synax_config import UPDATE_INTERVAL_MINUTES
from services.api.app.synax_ingestion_helper_functions import is_ingestion_enabled
from services.api.app.synax_ingestion_pipeline import (
    download_wikipedia_articles, download_arxiv_papers,
    download_clinicaltrials, download_pubmed_articles,
    download_wikidata_entities, download_openalex
)
from services.api.app.synax_wikipedia_arkiv_domain_keywords import wikipedia_keywords, arkiv_keywords as arxiv_keywords
from services.api.app.synax_clinicaltrials_pubmed_domain_keywords import clinical_trial_keywords, pubmed_keywords
from services.api.app.synax_wikidata_openalex_domain_keywords import wikidata_keywords, openalex_keywords
from services.api.app.synax_observability import log_event

async def run_ingestion_pipeline():
    while True:
        if not await is_ingestion_enabled():
            log_event("ingestion_skipped", status="skipped", reason="disabled_in_redis")
            await asyncio.sleep(UPDATE_INTERVAL_MINUTES * 60)
            continue

        try:
            await download_wikipedia_articles(wikipedia_keywords)
            log_event("wikipedia_ingestion_completed")
        except Exception as exc:
            log_event("wikipedia_ingestion_failed", status="failed", error=str(exc))

        try:
            results = await asyncio.gather(
                *(download_arxiv_papers(d, k) for d, kws in arxiv_keywords.items() for k in kws),
                return_exceptions=True
            )
            for r in results:
                if isinstance(r, Exception):
                    log_event("arxiv_item_failed", status="failed", error=str(r))
            log_event("arxiv_ingestion_completed")
        except Exception as exc:
            log_event("arxiv_ingestion_failed", status="failed", error=str(exc))

        try:
            await download_clinicaltrials(clinical_trial_keywords)
            log_event("clinicaltrials_ingestion_completed")
        except Exception as exc:
            log_event("clinicaltrials_ingestion_failed", status="failed", error=str(exc))

        try:
            await download_pubmed_articles(pubmed_keywords)
            log_event("pubmed_ingestion_completed")
        except Exception as exc:
            log_event("pubmed_ingestion_failed", status="failed", error=str(exc))

        try:
            await download_wikidata_entities(wikidata_keywords)
            log_event("wikidata_ingestion_completed")
        except Exception as exc:
            log_event("wikidata_ingestion_failed", status="failed", error=str(exc))

        try:
            await download_openalex(openalex_keywords)
            log_event("openalex_ingestion_completed")
        except Exception as exc:
            log_event("openalex_ingestion_failed", status="failed", error=str(exc))

        await asyncio.sleep(UPDATE_INTERVAL_MINUTES * 60)