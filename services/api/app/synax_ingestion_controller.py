# services/api/app/synax_ingestion_controller.py

from fastapi import APIRouter
from pydantic import BaseModel

from services.api.app.synax_ingestion_helper_functions import (
    is_ingestion_enabled,
    set_ingestion_enabled,
)
from services.api.app.synax_ingestion_pipeline_orchestrator import (
    request_ingestion_start,
)
from services.api.app.synax_observability import log_event


router = APIRouter(
    prefix="/admin/ingestion",
    tags=["Ingestion"],
)


class ToggleRequest(BaseModel):
    enabled: bool


@router.get("/status")
async def get_status():
    enabled = await is_ingestion_enabled()

    return {
        "enabled": enabled,
    }


@router.post("/toggle")
async def toggle(body: ToggleRequest):
    await set_ingestion_enabled(body.enabled)

    if body.enabled:
        await request_ingestion_start()

        log_event(
            "ingestion_enabled",
            status="enabled",
            source="redis",
        )

        return {
            "enabled": True,
            "started": True,
        }

    log_event(
        "ingestion_disabled",
        status="disabled",
        source="redis",
    )

    return {
        "enabled": False,
        "started": False,
    }