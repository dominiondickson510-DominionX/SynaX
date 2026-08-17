# services/api/app/synax_ingestion_pipeline_orchestrator.py

import asyncio

from services.api.app.synax_config import UPDATE_INTERVAL_MINUTES
from services.api.app.synax_ingestion_pipeline import (
    download_wikipedia_articles,
    download_arxiv_papers,
    download_clinicaltrials,
    download_pubmed_articles,
    download_wikidata_entities,
    download_openalex,
)
from services.api.app.synax_wikipedia_arxiv_domain_keywords import (
    wikipedia_keywords,
    arxiv_keywords,
)
from services.api.app.synax_clinicaltrials_pubmed_domain_keywords import (
    clinical_trial_keywords,
    pubmed_keywords,
)
from services.api.app.synax_wikidata_openalex_domain_keywords import (
    wikidata_keywords,
    openalex_keywords,
)
from services.api.app.synax_knowledge_graph import run_knowledge_graph
from services.api.app.synax_gpu_controller import run_gpu_job
from services.api.app.synax_ingestion_helper_functions import is_ingestion_enabled
from services.api.app.synax_observability import log_event

ingestion_wakeup_event = asyncio.Event()


async def request_ingestion_start() -> None:
    ingestion_wakeup_event.set()


async def _ingestion_allowed() -> bool:
    enabled = await is_ingestion_enabled()

    if not enabled:
        log_event(
            "ingestion_blocked",
            status="disabled",
            source="redis",
        )

    return enabled


async def _run_sync_stage(
    stage_name: str,
    stage,
) -> bool:
    if not await _ingestion_allowed():
        return False

    try:
        await asyncio.to_thread(stage)

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        log_event(
            f"{stage_name}_failed",
            status="failed",
            error=str(exc),
        )
        raise

    log_event(
        f"{stage_name}_completed",
        status="success",
    )

    return True


async def _run_source_ingestion() -> bool:
    if not await _ingestion_allowed():
        return False

    try:
        await download_wikipedia_articles(wikipedia_keywords)

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        log_event(
            "wikipedia_ingestion_failed",
            status="failed",
            error=str(exc),
        )
        raise

    log_event(
        "wikipedia_ingestion_completed",
        status="success",
    )

    if not await _ingestion_allowed():
        return False

    try:
        await asyncio.gather(
            *(
                download_arxiv_papers(
                    domain,
                    keyword,
                )
                for domain, keywords in arxiv_keywords.items()
                for keyword in keywords
            )
        )

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        log_event(
            "arxiv_ingestion_failed",
            status="failed",
            error=str(exc),
        )
        raise

    log_event(
        "arxiv_ingestion_completed",
        status="success",
    )

    if not await _ingestion_allowed():
        return False

    try:
        await download_clinicaltrials(clinical_trial_keywords)

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        log_event(
            "clinicaltrials_ingestion_failed",
            status="failed",
            error=str(exc),
        )
        raise

    log_event(
        "clinicaltrials_ingestion_completed",
        status="success",
    )

    if not await _ingestion_allowed():
        return False

    try:
        await download_pubmed_articles(pubmed_keywords)

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        log_event(
            "pubmed_ingestion_failed",
            status="failed",
            error=str(exc),
        )
        raise

    log_event(
        "pubmed_ingestion_completed",
        status="success",
    )

    if not await _ingestion_allowed():
        return False

    try:
        await download_wikidata_entities(wikidata_keywords)

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        log_event(
            "wikidata_ingestion_failed",
            status="failed",
            error=str(exc),
        )
        raise

    log_event(
        "wikidata_ingestion_completed",
        status="success",
    )

    if not await _ingestion_allowed():
        return False

    try:
        await download_openalex(openalex_keywords)

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        log_event(
            "openalex_ingestion_failed",
            status="failed",
            error=str(exc),
        )
        raise

    log_event(
        "openalex_ingestion_completed",
        status="success",
    )

    return True


async def run_ingestion_cycle() -> None:
    if not await _ingestion_allowed():
        return

    log_event(
        "ingestion_cycle_started",
        status="started",
    )

    if not await _run_source_ingestion():
        return

    if not await _ingestion_allowed():
        return

    log_event(
        "gpu_processing_requested",
        status="started",
        stages=[
            "entity_extraction",
            "coref_reso_entity_linking",
            "relationship_extraction",
            "embedding_generation",
        ],
    )

    try:
        gpu_result = await run_gpu_job()
    except asyncio.CancelledError:
        log_event(
            "gpu_processing_cancelled",
            status="cancelled",
        )
        raise
    except Exception as exc:
        log_event(
            "gpu_processing_failed",
            status="failed",
            error=str(exc),
        )
        raise

    if gpu_result != "COMPLETED":
        log_event(
            "gpu_processing_incomplete",
            status="failed",
            result=gpu_result,
        )
        return

    log_event(
        "gpu_processing_completed",
        status="success",
    )

    if not await _ingestion_allowed():
        return

    if not await _run_sync_stage(
        "knowledge_graph",
        run_knowledge_graph,
    ):
        return

    if not await _ingestion_allowed():
        return

    log_event(
        "ingestion_cycle_completed",
        status="success",
    )


async def _wait_for_next_ingestion_cycle() -> None:
    wake_task = asyncio.create_task(
        ingestion_wakeup_event.wait(),
        name="synax-ingestion-wakeup",
    )

    timer_task = asyncio.create_task(
        asyncio.sleep(UPDATE_INTERVAL_MINUTES * 60),
        name="synax-ingestion-timer",
    )

    try:
        done, pending = await asyncio.wait(
            {
                wake_task,
                timer_task,
            },
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        await asyncio.gather(
            *pending,
            return_exceptions=True,
        )

        if wake_task in done:
            log_event(
                "ingestion_pipeline_woken",
                status="manual_start",
            )

        elif timer_task in done:
            log_event(
                "ingestion_pipeline_interval_reached",
                status="scheduled",
                interval_minutes=UPDATE_INTERVAL_MINUTES,
            )

    except asyncio.CancelledError:
        wake_task.cancel()
        timer_task.cancel()

        await asyncio.gather(
            wake_task,
            timer_task,
            return_exceptions=True,
        )

        raise


async def run_ingestion_pipeline() -> None:
    log_event(
        "ingestion_pipeline_started",
        status="started",
        update_interval_minutes=UPDATE_INTERVAL_MINUTES,
    )

    while True:
        try:
            if await _ingestion_allowed():
                await run_ingestion_cycle()

            else:
                log_event(
                    "ingestion_cycle_skipped",
                    status="disabled",
                )

        except asyncio.CancelledError:
            log_event(
                "ingestion_pipeline_cancelled",
                status="shutdown",
            )
            raise

        except Exception as exc:
            log_event(
                "ingestion_pipeline_cycle_failed",
                status="failed",
                error=str(exc),
            )

        ingestion_wakeup_event.clear()

        await _wait_for_next_ingestion_cycle()