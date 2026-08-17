# services/api/app/synax_research_workspaces.py
import os
import asyncio
import uuid
import hashlib
import secrets
import jwt
import redis.asyncio as redis
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy import (
    Column,
    String,
    DateTime,
    ForeignKey,
    Index,
    func,
    select,
    text,
    Integer,
    Text,
    UniqueConstraint,
    Numeric,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "SYNAX_DB_URL", "postgresql+asyncpg://synax:synax@localhost/synax"
)
engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_size=5,
    max_overflow=5,
    pool_timeout=10,
    pool_recycle=300,
    pool_pre_ping=True,
)
AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)
Base = declarative_base()

redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable is required.")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXP_MINUTES = 180
REFRESH_TOKEN_EXP_DAYS = 14


class SynaXUser(Base):
    __tablename__ = "synax_users"
    user_id = Column(String, primary_key=True)
    email_address = Column(String, nullable=False, unique=True, index=True)
    user_attributes = Column(JSONB, nullable=False)
    password_hash = Column(String, nullable=False)
    credit_balance = Column(Numeric(12, 2), nullable=False, server_default=text("0"))
    workspace_limit = Column(Integer, nullable=False, server_default=text("1"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    workspaces = relationship("Workspace", back_populates="owner")


class PaymentTransaction(Base):
    __tablename__ = "payment_transactions"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("synax_users.user_id"), nullable=False, index=True)
    provider = Column(String, nullable=False)
    provider_reference = Column(String, nullable=False, unique=True, index=True)
    package_code = Column(String, nullable=False)
    package_name = Column(String, nullable=False)
    amount = Column(Integer, nullable=False)
    currency = Column(String, nullable=False)
    payment_status = Column(String, nullable=False, server_default=text("'pending'"))
    credits_granted = Column(Numeric(12, 2), nullable=False, server_default=text("0"))
    provider_transaction_id = Column(String, nullable=True)
    provider_status = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    verified_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "provider_reference",
            name="uq_payment_transaction_provider_reference",
        ),
    )


class CreditLedger(Base):
    __tablename__ = "credit_ledger"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("synax_users.user_id"), nullable=False, index=True)
    payment_transaction_id = Column(
        String, ForeignKey("payment_transactions.id"), nullable=True, index=True
    )
    research_history_id = Column(
        String, ForeignKey("research_history.id"), nullable=True, index=True
    )
    entry_type = Column(String, nullable=False)
    credits = Column(Numeric(12, 2), nullable=False)
    balance_after = Column(Numeric(12, 2), nullable=False)
    description = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    research_history = relationship("ResearchHistory", back_populates="credit_ledger")

    __table_args__ = (
        Index("ix_credit_ledger_user_created", "user_id", "created_at"),
    )


class Workspace(Base):
    __tablename__ = "workspaces"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("synax_users.user_id"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    title = Column(String, nullable=False)
    state = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    owner = relationship("SynaXUser", back_populates="workspaces")
    research_history = relationship(
        "ResearchHistory",
        back_populates="workspace",
        cascade="all, delete-orphan",
        order_by="ResearchHistory.created_at.desc()",
    )


class ResearchHistory(Base):
    __tablename__ = "research_history"
    id = Column(String, primary_key=True)
    workspace_id = Column(String, ForeignKey("workspaces.id"), nullable=False)
    query = Column(String, nullable=False)
    result = Column(JSONB, nullable=False)
    memory_sync_status = Column(
        String, nullable=False, server_default=text("'pending'")
    )
    memory_provider = Column(String, nullable=True)
    memory_id = Column(String, nullable=True)
    memory_sync_error = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    memory_synced_at = Column(DateTime(timezone=True), nullable=True)
    workspace = relationship("Workspace", back_populates="research_history")
    credit_ledger = relationship("CreditLedger", back_populates="research_history")

    __table_args__ = (
        Index("ix_research_history_workspace_created", "workspace_id", "created_at"),
        Index("ix_research_history_memory_sync_status", "memory_sync_status"),
    )


class PipelineState(Base):
    __tablename__ = "pipeline_state"
    source = Column(String, primary_key=True)
    external_id = Column(String, primary_key=True)
    filename = Column(String)
    meta = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    token_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("synax_users.user_id"), nullable=False)
    token_hash = Column(String, nullable=False, unique=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


async def get_session():
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


@asynccontextmanager
async def get_db():
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def generate_user_id(full_name: str) -> str:
    seed = secrets.token_bytes(16)
    raw = hashlib.sha256(seed + full_name.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=raw[:16]))


async def hash_password(password: str) -> str:
    return await asyncio.to_thread(pwd_context.hash, password)


async def verify_password(password: str, hashed: str) -> bool:
    return await asyncio.to_thread(pwd_context.verify, password, hashed)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    iat = int(now.timestamp())
    exp = int((now + timedelta(minutes=ACCESS_TOKEN_EXP_MINUTES)).timestamp())
    return jwt.encode(
        {
            "user_id": user_id,
            "type": "access",
            "iat": iat,
            "exp": exp,
            "jti": secrets.token_hex(16),
        },
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def create_refresh_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    iat = int(now.timestamp())
    exp = int((now + timedelta(days=REFRESH_TOKEN_EXP_DAYS)).timestamp())
    return jwt.encode(
        {
            "user_id": user_id,
            "type": "refresh",
            "iat": iat,
            "exp": exp,
            "jti": secrets.token_hex(16),
        },
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


async def persist_refresh_token(
    session: AsyncSession, user_id: str, refresh_token: str
):
    session.add(
        RefreshToken(
            token_id=str(uuid.uuid4()),
            user_id=user_id,
            token_hash=hash_token(refresh_token),
            expires_at=datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXP_DAYS),
        )
    )
    await session.flush()


async def revoke_access_token(jti: str, exp_timestamp: int):
    now = datetime.now(timezone.utc).timestamp()
    ttl = int(exp_timestamp - now)
    if ttl > 0:
        await redis_client.setex(f"blacklist:{jti}", ttl, "revoked")


async def is_token_revoked(jti: str) -> bool:
    return bool(await redis_client.exists(f"blacklist:{jti}"))


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid access token")

    jti = payload.get("jti")
    user_id = payload.get("user_id")
    exp = payload.get("exp")

    if not jti or not user_id or not exp:
        raise HTTPException(status_code=401, detail="Invalid token")

    if await is_token_revoked(jti):
        raise HTTPException(status_code=401, detail="Token revoked")

    return payload


async def get_ingestion_pipeline_state(
    session: AsyncSession, source: str, external_id: str
):
    result = await session.execute(
        select(PipelineState).where(
            PipelineState.source == source, PipelineState.external_id == external_id
        )
    )
    return result.scalar_one_or_none()


async def upsert_ingestion_pipeline_state(
    session: AsyncSession,
    source: str,
    external_id: str,
    filename: str | None = None,
    meta: dict | None = None,
):
    stmt = insert(PipelineState).values(
        source=source, external_id=external_id, filename=filename, meta=meta or {}
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["source", "external_id"],
        set_={
            "filename": stmt.excluded.filename,
            "meta": stmt.excluded.meta,
            "updated_at": func.now(),
        },
    )
    await session.execute(stmt)


router = APIRouter(prefix="/synax")


@router.post("/user", tags=["Onboarding"])
async def add_user(
    full_name: str = Form(...),
    email_address: str = Form(...),
    password: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    normalized_email = email_address.strip().lower()
    user_id = generate_user_id(full_name)
    user_attributes = {"full_name": full_name}

    existing = await session.execute(
        select(SynaXUser.user_id).where(SynaXUser.email_address == normalized_email)
    )
    if existing.scalars().first():
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = SynaXUser(
        user_id=user_id,
        email_address=normalized_email,
        user_attributes=user_attributes,
        password_hash=await hash_password(password),
        workspace_limit=2,
    )
    try:
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        return {
            "user_id": new_user.user_id,
            "user_attributes": new_user.user_attributes,
        }
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=400, detail="User already exists")
    except Exception:
        await session.rollback()
        raise HTTPException(status_code=500, detail="User creation failed")


@router.post("/login", tags=["Authentication"])
async def login(
    email_address: str = Form(...),
    password: str = Form(...),
    session: AsyncSession = Depends(get_session),
):
    email_address = email_address.strip().lower()
    user = (
        await session.execute(
            select(SynaXUser).where(SynaXUser.email_address == email_address)
        )
    ).scalars().first()

    if not user or not await verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(user.user_id)
    refresh_token = create_refresh_token(user.user_id)
    await persist_refresh_token(session, user.user_id, refresh_token)
    await session.commit()

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@router.post("/refresh", tags=["Token Management"])
async def refresh_token_endpoint(
    refresh_token: str = Form(...), session: AsyncSession = Depends(get_session)
):
    try:
        payload = jwt.decode(
            refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    token_hash = hash_token(refresh_token)
    stored_token = (
        await session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
    ).scalars().first()

    if not stored_token:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if stored_token.expires_at <= datetime.now(timezone.utc):
        await session.delete(stored_token)
        await session.commit()
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user_id = stored_token.user_id
    await session.delete(stored_token)

    new_access = create_access_token(user_id)
    new_refresh = create_refresh_token(user_id)
    await persist_refresh_token(session, user_id, new_refresh)
    await session.commit()

    return {
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "bearer",
    }


@router.post("/logout", tags=["Session Management"])
async def logout(
    request: Request,
    refresh_token: str = Form(...),
    user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    stored_token = (
        await session.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == hash_token(refresh_token),
                RefreshToken.user_id == user["user_id"],
            )
        )
    ).scalars().first()

    if stored_token:
        await session.delete(stored_token)
        await session.commit()

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        access_token = auth_header.split(" ", 1)[1]
        try:
            payload = jwt.decode(
                access_token,
                JWT_SECRET,
                algorithms=[JWT_ALGORITHM],
                options={"verify_exp": False},
            )
            if payload.get("type") == "access":
                jti = payload.get("jti")
                exp = payload.get("exp")
                if jti and exp:
                    await revoke_access_token(jti, exp)
        except jwt.InvalidTokenError:
            pass

    return {"message": "Logged out successfully"}


@router.post("/workspace", tags=["Workspace"])
async def create_workspace(
    title: str = Form(...),
    user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    title = title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Workspace title is required.")

    account = (
        await session.execute(
            select(SynaXUser)
            .where(SynaXUser.user_id == user["user_id"])
            .with_for_update()
        )
    ).scalar_one_or_none()

    if account is None:
        raise HTTPException(status_code=404, detail="User not found.")

    workspace_count = (
        await session.execute(
            select(func.count(Workspace.id)).where(
                Workspace.user_id == account.user_id
            )
        )
    ).scalar_one()

    if workspace_count >= account.workspace_limit:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "Workspace Limit Reached",
                "workspace_limit": account.workspace_limit,
                "workspace_count": workspace_count,
            },
        )

    workspace_id = str(uuid.uuid4())
    new_workspace = Workspace(
        id=workspace_id, user_id=account.user_id, title=title, state={}
    )
    session.add(new_workspace)
    await session.commit()
    await session.refresh(new_workspace)

    return {
        "workspace_id": workspace_id,
        "title": new_workspace.title,
        "state": new_workspace.state,
        "created_at": new_workspace.created_at,
    }


@router.get("/workspaces", tags=["Workspace"])
async def list_workspaces(
    user=Depends(get_current_user), session: AsyncSession = Depends(get_session)
):
    result = await session.execute(
        select(Workspace)
        .where(Workspace.user_id == user["user_id"])
        .order_by(Workspace.created_at.desc())
    )
    workspaces = result.scalars().all()

    return {
        "workspaces": [
            {
                "workspace_id": ws.id,
                "title": ws.title,
                "state": ws.state,
                "created_at": ws.created_at,
            }
            for ws in workspaces
        ]
    }


@router.get("/workspace/{workspace_id}/history", tags=["Research History"])
async def list_research_history(
    workspace_id: str,
    user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    workspace = (
        await session.execute(
            select(Workspace).where(
                Workspace.id == workspace_id, Workspace.user_id == user["user_id"]
            )
        )
    ).scalar_one_or_none()

    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    result = await session.execute(
        select(
            ResearchHistory.id, ResearchHistory.query, ResearchHistory.created_at
        )
        .where(ResearchHistory.workspace_id == workspace_id)
        .order_by(ResearchHistory.created_at.desc())
    )
    history = result.all()

    return {
        "workspace_id": workspace_id,
        "history": [
            {"history_id": history_id, "query": query, "created_at": created_at}
            for history_id, query, created_at in history
        ],
    }


@router.get("/workspace/{workspace_id}/history/{history_id}", tags=["Research History"])
async def get_research_history(
    workspace_id: str,
    history_id: str,
    user=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    workspace = (
        await session.execute(
            select(Workspace).where(
                Workspace.id == workspace_id, Workspace.user_id == user["user_id"]
            )
        )
    ).scalar_one_or_none()

    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    history = (
        await session.execute(
            select(ResearchHistory).where(
                ResearchHistory.id == history_id,
                ResearchHistory.workspace_id == workspace_id,
            )
        )
    ).scalar_one_or_none()

    if history is None:
        raise HTTPException(status_code=404, detail="Research history not found")

    return {
        "workspace_id": workspace_id,
        "history_id": history.id,
        "query": history.query,
        "result": history.result,
        "created_at": history.created_at,
    }