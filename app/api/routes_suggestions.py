from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.clients.data_backend_client import DataBackendClient
from app.core.settings import settings
from app.services.profile_service import ProfileService
from app.services.suggestion_agent_service import SuggestionAgentService
from app.services.suggestion_memory_service import SuggestionMemoryService
from app.services.suggestion_service import SuggestionService

router = APIRouter(tags=["suggestions"])

_data_backend_client = DataBackendClient()
_suggestion_service = SuggestionService(_data_backend_client)
_profile_service = ProfileService(_data_backend_client)
_suggestion_memory_service = SuggestionMemoryService(settings.database_url)
_suggestion_agent_service = SuggestionAgentService(
    _suggestion_service,
    _profile_service,
    _suggestion_memory_service,
)


def _validate_suggestions_payload(payload: dict[str, Any]) -> None:
    required_fields = ["occasion_details"]
    missing = [field for field in required_fields if not str(payload.get(field, "")).strip()]
    if missing:
        raise HTTPException(status_code=400, detail=f"Campos obrigatorios ausentes: {', '.join(missing)}")

    if "suggestions" not in payload:
        return

    if not isinstance(payload.get("suggestions"), list):
        raise HTTPException(status_code=400, detail="Campo suggestions deve ser lista")

    for index, item in enumerate(payload.get("suggestions", [])):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"Item suggestions[{index}] deve ser objeto")
        if not str(item.get("title", "")).strip():
            raise HTTPException(status_code=400, detail=f"Campo obrigatorio ausente: suggestions[{index}].title")
        if not str(item.get("reason", "")).strip():
            raise HTTPException(status_code=400, detail=f"Campo obrigatorio ausente: suggestions[{index}].reason")


@router.post(
    "/profiles/{friend_id}/suggestions",
    summary="Criar sugestões iniciais",
    description=(
        "Cria sugestões iniciais (batch/on-demand) com base no contexto do profile e detalhes da ocasião."
    ),
    responses={
        200: {"description": "Sugestões criadas"},
        400: {"description": "Payload inválido"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["occasion_details"],
                        "properties": {
                            "occasion_details": {"type": "string"},
                            "reminder_id": {"type": "string", "format": "uuid", "nullable": True},
                            "suggestions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["title", "reason"],
                                    "properties": {
                                        "title": {"type": "string"},
                                        "reason": {"type": "string"},
                                        "price_range": {"type": "string"},
                                        "tags": {"type": "array", "items": {"type": "string"}},
                                    },
                                },
                            },
                        },
                    },
                    "example": {
                        "occasion_details": "Aniversário de 30 anos, gosta de tecnologia",
                        "reminder_id": "32aab795-a9f8-4e25-a4b7-6fd080c97a18",
                    },
                }
            },
        }
    },
)
async def create_profile_suggestions(friend_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    _validate_suggestions_payload(payload)
    try:
        return await _suggestion_agent_service.create_initial_suggestions(friend_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/suggestions/agent/chat",
    summary="Conversar com suggestion agent",
    description=(
        "Continua a conversa de refinamento de uma sugestão específica usando gift_id como chave de sessão."
    ),
    responses={
        200: {"description": "Mensagem do agente retornada"},
        400: {"description": "Payload inválido"},
        404: {"description": "Sessão/gift não encontrado"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["gift_id", "message"],
                        "properties": {
                            "gift_id": {"type": "string", "format": "uuid"},
                            "message": {"type": "string"},
                            "friend_id": {"type": "string", "format": "uuid"},
                            "occasion_details": {"type": "string"},
                        },
                    },
                    "example": {
                        "gift_id": "a6501206-411e-4a34-8217-bf35974f86e9",
                        "message": "Quero algo mais personalizado e até R$200",
                    },
                }
            },
        }
    },
)
async def chat_suggestion_agent(request: Request) -> dict[str, Any]:
    payload = await request.json()
    gift_id = str(payload.get("gift_id", "")).strip()
    message = str(payload.get("message", "")).strip()
    friend_id = str(payload.get("friend_id", "")).strip()
    occasion_details = str(payload.get("occasion_details", "")).strip()

    missing_fields = [
        field
        for field, value in {
            "gift_id": gift_id,
            "message": message,
        }.items()
        if not value
    ]
    if missing_fields:
        raise HTTPException(status_code=400, detail=f"Campos obrigatorios ausentes: {', '.join(missing_fields)}")

    try:
        return await _suggestion_agent_service.chat(
            gift_id=gift_id,
            user_message=message,
            friend_id=friend_id or None,
            occasion_details=occasion_details,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/suggestions/agent/finalize",
    summary="Finalizar refinamento de sugestão",
    description="Consolida histórico da conversa e persiste a sugestão refinada no gift existente.",
    responses={
        200: {"description": "Sugestão refinada persistida"},
        400: {"description": "Payload inválido"},
        404: {"description": "Sessão não encontrada"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["gift_id"],
                        "properties": {
                            "gift_id": {"type": "string", "format": "uuid"},
                            "friend_id": {"type": "string", "format": "uuid"},
                        },
                    },
                    "example": {
                        "gift_id": "a6501206-411e-4a34-8217-bf35974f86e9",
                        "friend_id": "c096d057-a0cf-4c2d-857b-41beccb42de8",
                    },
                }
            },
        }
    },
)
async def finalize_suggestion_agent(request: Request) -> dict[str, Any]:
    payload = await request.json()
    gift_id = str(payload.get("gift_id", "")).strip()
    friend_id = str(payload.get("friend_id", "")).strip()
    if not gift_id:
        raise HTTPException(status_code=400, detail="Campo obrigatorio ausente: gift_id")

    try:
        return await _suggestion_agent_service.finalize_suggestion(gift_id=gift_id, friend_id=friend_id or None)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail == "Sessao de sugestao nao encontrada" else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc


