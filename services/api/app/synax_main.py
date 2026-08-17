# services/api/app/synax_main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.app.synax_lifespan import lifespan
from services.api.app.synax_research_workspaces import router as workspace_router
from services.api.app.synax_research_suggestions import router as suggestions_router
from services.api.app.synax_billing import router as billing_router
from services.api.app.synax_query import router as query_router
from services.api.app.synax_ingestion_controller import router as ingestion_router


app = FastAPI(
    title="SynaX API",
    description="SynaX AI Research Operating System API",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(workspace_router)
app.include_router(suggestions_router)
app.include_router(billing_router)
app.include_router(query_router)
app.include_router(ingestion_router)


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "service": "synax-api",
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "SynaX",
        "service": "AI Research Operating System",
        "status": "online",
    }