# services/api/app/synax_gpu_worker.py
import asyncio
import os
import traceback

import torch
from dotenv import load_dotenv

from services.api.app.synax_config import redis_client
from services.api.app.synax_entity_extraction import run_entity_extraction
from services.api.app.synax_coref_reso_entity_linking import run_coref_reso_entity_linking
from services.api.app.synax_relationship_extraction import run_relationship_extraction
from services.api.app.synax_embedding_generation import run_embedding_generation
from services.api.app.synax_observability import log_event

load_dotenv()

GPU_JOB_ID = os.getenv("SYNAX_GPU_JOB_ID")

if not GPU_JOB_ID:
    raise RuntimeError("SYNAX_GPU_JOB_ID is required.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on the GPU worker.")


def _job_status_key() -> str:
    return f"synax:gpu:job:{GPU_JOB_ID}:status"


def _job_error_key() -> str:
    return f"synax:gpu:job:{GPU_JOB_ID}:error"


async def _set_job_status(
    status: str,
    error: str | None = None,
) -> None:
    await redis_client.set(_job_status_key(), status)

    if error:
        await redis_client.set(_job_error_key(), error)
    else:
        await redis_client.delete(_job_error_key())


async def _run_stage(stage_name: str, stage) -> None:
    log_event(
        f"{stage_name}_started",
        status="started",
        execution_target="gpu",
        job_id=GPU_JOB_ID,
    )

    try:
        await asyncio.to_thread(stage)

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        log_event(
            f"{stage_name}_failed",
            status="failed",
            execution_target="gpu",
            job_id=GPU_JOB_ID,
            error=str(exc),
        )
        raise

    log_event(
        f"{stage_name}_completed",
        status="success",
        execution_target="gpu",
        job_id=GPU_JOB_ID,
    )


async def run_gpu_pipeline() -> None:
    await _set_job_status("RUNNING")

    log_event(
        "gpu_device_detected",
        status="success",
        execution_target="gpu",
        job_id=GPU_JOB_ID,
        device=torch.cuda.get_device_name(0),
    )

    log_event(
        "gpu_pipeline_started",
        status="started",
        execution_target="gpu",
        job_id=GPU_JOB_ID,
    )

    try:
        await _run_stage(
            "entity_extraction",
            run_entity_extraction,
        )

        await _run_stage(
            "coref_reso_entity_linking",
            run_coref_reso_entity_linking,
        )

        await _run_stage(
            "relationship_extraction",
            run_relationship_extraction,
        )

        await _run_stage(
            "embedding_generation",
            run_embedding_generation,
        )

    except asyncio.CancelledError:
        await _set_job_status("CANCELLED")

        log_event(
            "gpu_pipeline_cancelled",
            status="cancelled",
            execution_target="gpu",
            job_id=GPU_JOB_ID,
        )

        raise

    except Exception as exc:
        error = traceback.format_exc()

        await _set_job_status(
            "FAILED",
            error=error,
        )

        log_event(
            "gpu_pipeline_failed",
            status="failed",
            execution_target="gpu",
            job_id=GPU_JOB_ID,
            error=str(exc),
            traceback=error,
        )

        raise

    await _set_job_status("COMPLETED")

    log_event(
        "gpu_pipeline_completed",
        status="success",
        execution_target="gpu",
        job_id=GPU_JOB_ID,
    )


if __name__ == "__main__":
    asyncio.run(run_gpu_pipeline())