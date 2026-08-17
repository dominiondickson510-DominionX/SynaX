# services/api/app/synax_gpu_controller.py
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv

from google.cloud import compute_v1

from services.api.app.synax_config import redis_client
from services.api.app.synax_observability import log_event

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_GPU_ZONE = os.getenv("GCP_GPU_ZONE", "europe-west1-b")
GCP_GPU_VM_NAME = os.getenv("GCP_GPU_VM_NAME", "synax-server")

GPU_START_TIMEOUT_SECONDS = int(
    os.getenv("GPU_START_TIMEOUT_SECONDS", "300")
)

GPU_MAX_RUNTIME_SECONDS = int(
    os.getenv("GPU_MAX_RUNTIME_SECONDS", "7200")
)

GPU_POLL_INTERVAL_SECONDS = int(
    os.getenv("GPU_POLL_INTERVAL_SECONDS", "5")
)

if not GCP_PROJECT_ID:
    raise RuntimeError("GCP_PROJECT_ID is required.")

_instances_client = compute_v1.InstancesClient()
_operations_client = compute_v1.ZoneOperationsClient()


def _job_status_key(job_id: str) -> str:
    return f"synax:gpu:job:{job_id}:status"


def _job_error_key(job_id: str) -> str:
    return f"synax:gpu:job:{job_id}:error"


async def _get_vm():
    return await asyncio.to_thread(
        _instances_client.get,
        project=GCP_PROJECT_ID,
        zone=GCP_GPU_ZONE,
        instance=GCP_GPU_VM_NAME,
    )


async def get_gpu_vm_status() -> str:
    instance = await _get_vm()
    return instance.status


async def _wait_for_operation(operation_name: str) -> None:
    while True:
        operation = await asyncio.to_thread(
            _operations_client.get,
            project=GCP_PROJECT_ID,
            zone=GCP_GPU_ZONE,
            operation=operation_name,
        )

        if operation.status == compute_v1.Operation.Status.DONE:
            if operation.error:
                raise RuntimeError(str(operation.error))
            return

        await asyncio.sleep(GPU_POLL_INTERVAL_SECONDS)


async def set_gpu_job_metadata(job_id: str) -> None:
    instance = await _get_vm()

    metadata = instance.metadata

    if metadata is None:
        metadata = compute_v1.Metadata()

    items = list(metadata.items)

    job_item = None

    for item in items:
        if item.key == "SYNAX_GPU_JOB_ID":
            job_item = item
            break

    if job_item:
        job_item.value = job_id
    else:
        items.append(
            compute_v1.Items(
                key="SYNAX_GPU_JOB_ID",
                value=job_id,
            )
        )

    metadata.items = items

    operation = await asyncio.to_thread(
        _instances_client.set_metadata,
        project=GCP_PROJECT_ID,
        zone=GCP_GPU_ZONE,
        instance=GCP_GPU_VM_NAME,
        metadata_resource=metadata,
    )

    await _wait_for_operation(operation.name)

    log_event(
        "gpu_job_metadata_set",
        status="success",
        job_id=job_id,
        vm=GCP_GPU_VM_NAME,
    )


async def start_gpu_vm(job_id: str) -> None:
    status = await get_gpu_vm_status()

    if status == "RUNNING":
        raise RuntimeError(
            f"GPU VM {GCP_GPU_VM_NAME} is already running."
        )

    if status not in {"TERMINATED", "STOPPED"}:
        raise RuntimeError(
            f"GPU VM cannot be started from state: {status}"
        )

    await set_gpu_job_metadata(job_id)

    log_event(
        "gpu_vm_start_requested",
        status="starting",
        job_id=job_id,
        vm=GCP_GPU_VM_NAME,
        zone=GCP_GPU_ZONE,
    )

    operation = await asyncio.to_thread(
        _instances_client.start,
        project=GCP_PROJECT_ID,
        zone=GCP_GPU_ZONE,
        instance=GCP_GPU_VM_NAME,
    )

    await _wait_for_operation(operation.name)

    log_event(
        "gpu_vm_start_completed",
        status="started",
        job_id=job_id,
        vm=GCP_GPU_VM_NAME,
    )


async def wait_for_gpu_vm_running() -> None:
    started_at = time.monotonic()

    while True:
        status = await get_gpu_vm_status()

        if status == "RUNNING":
            log_event(
                "gpu_vm_running",
                status="running",
                vm=GCP_GPU_VM_NAME,
            )
            return

        if time.monotonic() - started_at >= GPU_START_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"GPU VM did not reach RUNNING state within "
                f"{GPU_START_TIMEOUT_SECONDS} seconds."
            )

        await asyncio.sleep(GPU_POLL_INTERVAL_SECONDS)


async def wait_for_gpu_job(job_id: str) -> str:
    status_key = _job_status_key(job_id)
    error_key = _job_error_key(job_id)

    started_at = time.monotonic()

    while True:
        status = await redis_client.get(status_key)

        if status in {"COMPLETED", "FAILED", "CANCELLED"}:
            if status == "FAILED":
                error = await redis_client.get(error_key)

                log_event(
                    "gpu_job_failed",
                    status="failed",
                    job_id=job_id,
                    error=error,
                )

            else:
                log_event(
                    "gpu_job_finished",
                    status=status.lower(),
                    job_id=job_id,
                )

            return status

        if time.monotonic() - started_at >= GPU_MAX_RUNTIME_SECONDS:
            raise TimeoutError(
                f"GPU job {job_id} exceeded maximum runtime of "
                f"{GPU_MAX_RUNTIME_SECONDS} seconds."
            )

        await asyncio.sleep(GPU_POLL_INTERVAL_SECONDS)


async def stop_gpu_vm() -> None:
    status = await get_gpu_vm_status()

    if status in {"TERMINATED", "STOPPED"}:
        log_event(
            "gpu_vm_already_stopped",
            status="stopped",
            vm=GCP_GPU_VM_NAME,
        )
        return

    if status != "RUNNING":
        raise RuntimeError(
            f"GPU VM cannot be stopped from state: {status}"
        )

    log_event(
        "gpu_vm_stop_requested",
        status="stopping",
        vm=GCP_GPU_VM_NAME,
    )

    operation = await asyncio.to_thread(
        _instances_client.stop,
        project=GCP_PROJECT_ID,
        zone=GCP_GPU_ZONE,
        instance=GCP_GPU_VM_NAME,
    )

    await _wait_for_operation(operation.name)

    log_event(
        "gpu_vm_stop_completed",
        status="stopped",
        vm=GCP_GPU_VM_NAME,
    )


async def run_gpu_job() -> str:
    job_id = uuid.uuid4().hex

    status_key = _job_status_key(job_id)

    await redis_client.set(status_key, "PENDING")

    log_event(
        "gpu_job_created",
        status="pending",
        job_id=job_id,
        vm=GCP_GPU_VM_NAME,
    )

    vm_started = False

    try:
        await start_gpu_vm(job_id)
        vm_started = True

        await wait_for_gpu_vm_running()

        log_event(
            "gpu_job_vm_ready",
            status="ready",
            job_id=job_id,
            vm=GCP_GPU_VM_NAME,
        )

        result = await wait_for_gpu_job(job_id)

        if result != "COMPLETED":
            raise RuntimeError(
                f"GPU job {job_id} finished with status {result}."
            )

        return result

    finally:
        if vm_started:
            await stop_gpu_vm()

        await redis_client.delete(status_key)
        await redis_client.delete(_job_error_key(job_id))