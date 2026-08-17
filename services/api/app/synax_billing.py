# services/api/app/synax_billing.py
import hashlib
import hmac
import os
import uuid
import httpx
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Optional
from pydantic import BaseModel
from decimal import Decimal
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from services.api.app.synax_config import PAYSTACK_SECRET_KEY
from services.api.app.synax_config import PAYSTACK_BASE_URL
from services.api.app.synax_research_workspaces import SynaXUser
from services.api.app.synax_research_workspaces import PaymentTransaction
from services.api.app.synax_research_workspaces import CreditLedger
from services.api.app.synax_research_workspaces import Workspace
from services.api.app.synax_research_workspaces import get_session
from services.api.app.synax_research_workspaces import get_current_user
from services.api.app.synax_observability import log_event


@dataclass(frozen=True, slots=True)
class CreditPackage:
    code: str
    name: str
    credits: Decimal
    amount: int
    currency: str
    max_workspaces: int


CREDIT_PACKAGES = {
    "starter": CreditPackage(
        code="starter",
        name="Starter Credits",
        credits=Decimal("40"),
        amount=300000,
        currency="NGN",
        max_workspaces=2,
    ),
    "research": CreditPackage(
        code="research",
        name="Research Credits",
        credits=Decimal("100"),
        amount=700000,
        currency="NGN",
        max_workspaces=5,
    ),
    "frontier": CreditPackage(
        code="frontier",
        name="Frontier Credits",
        credits=Decimal("230"),
        amount=1500000,
        currency="NGN",
        max_workspaces=10,
    ),
}

QUERY_CREDIT_COST = Decimal("0.90")


class PaymentInitialization(BaseModel):
    package_code: str


class PaymentVerification(BaseModel):
    reference: str


@dataclass(frozen=True, slots=True)
class ProviderTransaction:
    reference: str
    status: str
    amount: int
    currency: str
    provider_transaction_id: Optional[str] = None


class PaymentProvider(ABC):
    @abstractmethod
    async def initialize(
        self,
        *,
        email: str,
        amount: int,
        currency: str,
        reference: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def verify(self, *, reference: str) -> ProviderTransaction:
        raise NotImplementedError


class PaystackProvider(PaymentProvider):
    def __init__(self, *, secret_key: str, base_url: str = PAYSTACK_BASE_URL):
        self.secret_key = secret_key
        self.base_url = base_url

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.secret_key}",
            "Content-Type": "application/json",
        }

    async def initialize(
        self,
        *,
        email: str,
        amount: int,
        currency: str,
        reference: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "email": email,
            "amount": amount,
            "currency": currency,
            "reference": reference,
            "metadata": metadata,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/transaction/initialize",
                headers=self._headers(),
                json=payload,
            )
        response.raise_for_status()
        body = response.json()
        if not body.get("status"):
            raise RuntimeError(body.get("message", "Paystack initialization failed."))
        data = body.get("data") or {}
        return {
            "authorization_url": data["authorization_url"],
            "access_code": data["access_code"],
            "reference": data["reference"],
        }

    async def verify(self, *, reference: str) -> ProviderTransaction:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/transaction/verify/{reference}",
                headers=self._headers(),
            )
        response.raise_for_status()
        body = response.json()
        if not body.get("status"):
            raise RuntimeError(body.get("message", "Paystack verification failed."))
        data = body.get("data") or {}
        return ProviderTransaction(
            reference=data.get("reference", reference),
            status=str(data.get("status", "")).lower(),
            amount=int(data.get("amount", 0)),
            currency=str(data.get("currency", "")).upper(),
            provider_transaction_id=(
                str(data["id"]) if data.get("id") is not None else None
            ),
        )


payment_provider: PaymentProvider = PaystackProvider(secret_key=PAYSTACK_SECRET_KEY)


def get_package(package_code: str) -> CreditPackage:
    package = CREDIT_PACKAGES.get(package_code)
    if package is None:
        raise HTTPException(status_code=400, detail="Invalid credit package.")
    return package


def generate_payment_reference() -> str:
    return f"synax_{uuid.uuid4().hex}"


async def initialize_payment_transaction(
    *,
    user: SynaXUser,
    package: CreditPackage,
    session: AsyncSession,
) -> dict[str, Any]:
    reference = generate_payment_reference()
    transaction = PaymentTransaction(
        id=str(uuid.uuid4()),
        user_id=user.user_id,
        provider="paystack",
        provider_reference=reference,
        package_code=package.code,
        package_name=package.name,
        amount=package.amount,
        currency=package.currency,
        payment_status="pending",
        credits_granted=0,
    )
    session.add(transaction)
    await session.flush()
    try:
        payment = await payment_provider.initialize(
            email=user.email_address,
            amount=package.amount,
            currency=package.currency,
            reference=reference,
            metadata={
                "user_id": user.user_id,
                "package_code": package.code,
                "credits": str(package.credits),
                "transaction_id": transaction.id,
            },
        )
    except Exception as exc:
        log_event(
            "payment_initialization_failed",
            status="failed",
            user_id=user.user_id,
            transaction_id=transaction.id,
            error=str(exc),
        )
        await session.rollback()
        raise
    await session.commit()
    log_event(
        "payment_initialization_completed",
        status="success",
        user_id=user.user_id,
        transaction_id=transaction.id,
        package_code=package.code,
    )
    return {
        "transaction_id": transaction.id,
        "reference": payment["reference"],
        "authorization_url": payment["authorization_url"],
        "access_code": payment["access_code"],
        "package": {
            "code": package.code,
            "name": package.name,
            "credits": package.credits,
            "amount": package.amount,
            "currency": package.currency,
        },
    }


async def fulfill_payment(
    *,
    transaction: PaymentTransaction,
    provider_transaction: ProviderTransaction,
    session: AsyncSession,
) -> bool:
    transaction = (
        await session.execute(
            select(PaymentTransaction)
            .where(PaymentTransaction.id == transaction.id)
            .with_for_update()
        )
    ).scalar_one()
    if transaction.payment_status == "completed":
        return False
    if provider_transaction.status != "success":
        transaction.payment_status = provider_transaction.status or "failed"
        transaction.provider_status = provider_transaction.status
        transaction.provider_transaction_id = provider_transaction.provider_transaction_id
        transaction.verified_at = datetime.now(timezone.utc)
        return False
    if provider_transaction.amount != transaction.amount:
        transaction.payment_status = "amount_mismatch"
        transaction.provider_status = provider_transaction.status
        transaction.provider_transaction_id = provider_transaction.provider_transaction_id
        transaction.verified_at = datetime.now(timezone.utc)
        raise ValueError("Payment amount does not match the SynaX package.")
    if provider_transaction.currency.upper() != transaction.currency.upper():
        transaction.payment_status = "currency_mismatch"
        transaction.provider_status = provider_transaction.status
        transaction.provider_transaction_id = provider_transaction.provider_transaction_id
        transaction.verified_at = datetime.now(timezone.utc)
        raise ValueError("Payment currency does not match the SynaX package.")
    user = (
        await session.execute(
            select(SynaXUser)
            .where(SynaXUser.user_id == transaction.user_id)
            .with_for_update()
        )
    ).scalar_one()
    existing_ledger = (
        await session.execute(
            select(CreditLedger)
            .where(CreditLedger.payment_transaction_id == transaction.id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if existing_ledger is not None:
        transaction.payment_status = "completed"
        transaction.credits_granted = existing_ledger.credits
        return False
    package = get_package(transaction.package_code)
    new_balance = Decimal(str(user.credit_balance)) + Decimal(package.credits)
    ledger = CreditLedger(
        id=str(uuid.uuid4()),
        user_id=user.user_id,
        payment_transaction_id=transaction.id,
        entry_type="purchase",
        credits=package.credits,
        balance_after=new_balance,
        description=f"Purchase of {package.name}",
    )
    user.credit_balance = new_balance
    user.workspace_limit = max(user.workspace_limit, package.max_workspaces)
    transaction.payment_status = "completed"
    transaction.provider_status = provider_transaction.status
    transaction.provider_transaction_id = provider_transaction.provider_transaction_id
    transaction.credits_granted = package.credits
    transaction.verified_at = datetime.now(timezone.utc)
    transaction.completed_at = datetime.now(timezone.utc)
    session.add(ledger)
    log_event(
        "payment_fulfilled",
        user_id=user.user_id,
        transaction_id=transaction.id,
        package_code=package.code,
        credits=package.credits,
        amount=transaction.amount,
        currency=transaction.currency,
    )
    return True


async def verify_and_fulfill_payment(
    *, reference: str, session: AsyncSession
) -> PaymentTransaction:
    transaction = (
        await session.execute(
            select(PaymentTransaction).where(
                PaymentTransaction.provider == "paystack",
                PaymentTransaction.provider_reference == reference,
            )
        )
    ).scalar_one_or_none()
    if transaction is None:
        raise HTTPException(status_code=404, detail="Payment transaction not found.")
    if transaction.payment_status == "completed":
        return transaction
    provider_transaction = await payment_provider.verify(reference=reference)
    if provider_transaction.reference != transaction.provider_reference:
        raise HTTPException(status_code=400, detail="Payment reference mismatch.")
    try:
        await fulfill_payment(
            transaction=transaction,
            provider_transaction=provider_transaction,
            session=session,
        )
        await session.commit()
    except Exception as exc:
        log_event("payment_verification_failed", reference=reference, error=str(exc))
        await session.rollback()
        raise
    await session.refresh(transaction)
    return transaction


def verify_paystack_signature(*, payload: bytes, signature: str) -> bool:
    expected = hmac.new(
        PAYSTACK_SECRET_KEY.encode("utf-8"), payload, hashlib.sha512
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


async def consume_credits(
    *,
    user_id: str,
    cost: Decimal,
    description: str,
    session: AsyncSession,
    research_history_id: Optional[str] = None,
) -> Decimal:
    user = (
        await session.execute(
            select(SynaXUser)
            .where(SynaXUser.user_id == user_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    balance = Decimal(str(user.credit_balance))
    cost = Decimal(str(cost))
    if balance < cost:
        log_event(
            "credit_consumption_rejected",
            user_id=user_id,
            required_credits=str(cost),
            available_credits=str(balance),
            description=description,
        )
        raise HTTPException(
            status_code=402,
            detail={
                "code": "Insufficient Credits",
                "required_credits": str(cost),
                "available_credits": str(balance),
                "shortfall": str(cost - balance),
            },
        )
    new_balance = balance - cost
    user.credit_balance = new_balance
    session.add(
        CreditLedger(
            id=str(uuid.uuid4()),
            user_id=user_id,
            payment_transaction_id=None,
            research_history_id=research_history_id,
            entry_type="consumption",
            credits=-cost,
            balance_after=new_balance,
            description=description,
        )
    )
    await session.flush()
    log_event(
        "credits_consumed",
        user_id=user_id,
        credits=str(cost),
        balance_after=str(new_balance),
        description=description,
        research_history_id=research_history_id,
    )
    return new_balance


async def refund_credits(
    *, user_id: str, amount: Decimal, description: str, session: AsyncSession
) -> Decimal:
    user = (
        await session.execute(
            select(SynaXUser)
            .where(SynaXUser.user_id == user_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    balance = Decimal(str(user.credit_balance))
    new_balance = balance + amount
    user.credit_balance = new_balance
    session.add(
        CreditLedger(
            id=str(uuid.uuid4()),
            user_id=user_id,
            payment_transaction_id=None,
            entry_type="refund",
            credits=amount,
            balance_after=new_balance,
            description=description,
        )
    )
    await session.flush()
    log_event(
        "credits_refunded",
        user_id=user_id,
        credits=str(amount),
        balance_after=str(new_balance),
        description=description,
    )
    return new_balance


router = APIRouter(prefix="/billing", tags=["Billing"])


@router.get("/packages")
async def list_credit_packages():
    packages = list(CREDIT_PACKAGES.values())
    return {
        "currency": packages[0].currency,
        "packages": [
            {
                "code": p.code,
                "name": p.name,
                "credits": str(p.credits),
                "amount": p.amount // 100,
                "currency": p.currency,
                "sessions": int(p.credits // QUERY_CREDIT_COST),
                "max_workspaces": p.max_workspaces,
            }
            for p in packages
        ],
    }


@router.post("/payment/initialize")
async def initialize_payment(
    request: PaymentInitialization,
    user_payload=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    user = (
        await session.execute(
            select(SynaXUser).where(SynaXUser.user_id == user_payload["user_id"])
        )
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    package = get_package(request.package_code)
    return await initialize_payment_transaction(
        user=user, package=package, session=session
    )


@router.post("/payment/verify")
async def verify_payment(
    request: PaymentVerification,
    user_payload=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    transaction = (
        await session.execute(
            select(PaymentTransaction).where(
                PaymentTransaction.provider_reference == request.reference,
                PaymentTransaction.user_id == user_payload["user_id"],
            )
        )
    ).scalar_one_or_none()
    if transaction is None:
        raise HTTPException(status_code=404, detail="Payment transaction not found.")
    transaction = await verify_and_fulfill_payment(reference=request.reference, session=session)
    return {
        "transaction_id": transaction.id,
        "reference": transaction.provider_reference,
        "payment_status": transaction.payment_status,
        "credits_granted": transaction.credits_granted,
        "user_id": transaction.user_id,
    }


@router.post("/webhook/paystack")
async def paystack_webhook(
    request: Request, session: AsyncSession = Depends(get_session)
):
    payload = await request.body()
    signature = request.headers.get("x-paystack-signature")
    if not signature or not verify_paystack_signature(payload=payload, signature=signature):
        raise HTTPException(status_code=401, detail="Invalid Paystack signature.")
    event = await request.json()
    log_event(
        "paystack_webhook_received",
        event=event.get("event"),
        reference=(event.get("data") or {}).get("reference"),
    )
    if event.get("event") != "charge.success":
        return {"status": "ignored"}
    data = event.get("data") or {}
    reference = data.get("reference")
    if not reference:
        return {"status": "ignored"}
    transaction = (
        await session.execute(
            select(PaymentTransaction).where(
                PaymentTransaction.provider == "paystack",
                PaymentTransaction.provider_reference == reference,
            )
        )
    ).scalar_one_or_none()
    if transaction is None:
        return {"status": "ignored"}
    if transaction.payment_status == "completed":
        return {"status": "already_processed"}
    await verify_and_fulfill_payment(reference=reference, session=session)
    log_event(
        "paystack_webhook_processed",
        reference=reference,
        transaction_id=transaction.id,
    )
    return {"status": "processed"}


@router.get("/account")
async def get_billing_account(
    user_payload=Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    account = (
        await session.execute(
            select(SynaXUser.credit_balance, SynaXUser.workspace_limit).where(
                SynaXUser.user_id == user_payload["user_id"]
            )
        )
    ).one_or_none()
    if account is None:
        raise HTTPException(status_code=404, detail="User not found.")
    credit_balance, workspace_limit = account
    workspace_count = (
        await session.execute(
            select(func.count(Workspace.id)).where(
                Workspace.user_id == user_payload["user_id"]
            )
        )
    ).scalar_one()
    return {
        "credits": str(Decimal(str(credit_balance))),
        "workspace_limit": workspace_limit,
        "workspace_count": workspace_count,
        "workspaces_remaining": max(0, workspace_limit - workspace_count),
    }