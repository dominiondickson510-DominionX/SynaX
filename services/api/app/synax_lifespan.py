# services/api/app/synax_lifespan.py

import asyncio
import inspect

from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from services.api.app.synax_config import (
    redis_client,
    FAISS_SEARCH_EXECUTOR,
    gpt_client,
    gemini_client,
    neo4j_driver,
    supermemory_client,
)
from services.api.app.synax_research_workspaces import (
    engine,
)
from services.api.app.synax_ingestion_pipeline_orchestrator import (
    run_ingestion_pipeline,
)
from services.api.app.synax_observability import log_event


async def _close_resource(resource, name: str):
    if resource is None:
        return

    try:
        close_method = getattr(resource, "aclose", None)

        if close_method is None:
            close_method = getattr(resource, "close", None)

        if close_method is None:
            return

        result = close_method()

        if inspect.isawaitable(result):
            await result

        log_event(
            f"{name}_closed",
            status="success",
        )

    except Exception as exc:
        log_event(
            f"{name}_close_failed",
            status="failed",
            error=str(exc),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    ingestion_task = None

    log_event(
        "synax_startup_started",
        status="started",
    )

    try:
        async with engine.begin() as connection:
            await connection.execute(text("SELECT 1"))

        log_event(
            "postgresql_connection_verified",
            status="success",
        )

        await redis_client.ping()

        log_event(
            "redis_connection_verified",
            status="success",
        )

        await asyncio.to_thread(
            neo4j_driver.verify_connectivity
        )

        log_event(
            "neo4j_connection_verified",
            status="success",
        )

        ingestion_task = asyncio.create_task(
            run_ingestion_pipeline(),
            name="synax-ingestion-pipeline",
        )

        app.state.ingestion_task = ingestion_task

        log_event(
            "ingestion_pipeline_task_started",
            status="success",
        )

        log_event(
            "synax_startup_completed",
            status="success",
        )

        yield

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        log_event(
            "synax_startup_failed",
            status="failed",
            error=str(exc),
        )
        raise

    finally:
        log_event(
            "synax_shutdown_started",
            status="started",
        )

        if ingestion_task is not None:
            ingestion_task.cancel()

            try:
                await ingestion_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                log_event(
                    "ingestion_pipeline_shutdown_failed",
                    status="failed",
                    error=str(exc),
                )

            log_event(
                "ingestion_pipeline_task_stopped",
                status="success",
            )

        try:
            FAISS_SEARCH_EXECUTOR.shutdown(
                wait=True,
                cancel_futures=True,
            )

            log_event(
                "faiss_executor_shutdown",
                status="success",
            )

        except Exception as exc:
            log_event(
                "faiss_executor_shutdown_failed",
                status="failed",
                error=str(exc),
            )

        await _close_resource(
            supermemory_client,
            "supermemory_client",
        )

        await _close_resource(
            gemini_client,
            "gemini_client",
        )

        await _close_resource(
            gpt_client,
            "gpt_client",
        )

        await _close_resource(
            redis_client,
            "redis_client",
        )

        try:
            neo4j_driver.close()

            log_event(
                "neo4j_driver_closed",
                status="success",
            )

        except Exception as exc:
            log_event(
                "neo4j_driver_close_failed",
                status="failed",
                error=str(exc),
            )

        try:
            await engine.dispose()

            log_event(
                "postgresql_engine_disposed",
                status="success",
            )

        except Exception as exc:
            log_event(
                "postgresql_engine_dispose_failed",
                status="failed",
                error=str(exc),
            )

        log_event(
            "synax_shutdown_completed",
            status="success",
        )