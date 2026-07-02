from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.clients.data_backend_client import DataBackendClient
from app.core.settings import settings
from app.services.profile_agent_service import ProfileAgentService
from app.services.profile_service import ProfileService
from app.services.embedding_service import EmbeddingService
from app.services.profile_memory_service import ProfileMemoryService
from app.clients.embedding_client import EmbeddingClient

router = APIRouter(prefix="/profiles", tags=["profiles"])

_data_backend_client = DataBackendClient()
_profile_service = ProfileService(_data_backend_client)
_embedding_service = EmbeddingService(_data_backend_client, EmbeddingClient())
_profile_memory_service = ProfileMemoryService(settings.database_url)
_profile_agent_service = ProfileAgentService(_profile_service, _embedding_service, _profile_memory_service)


def _read_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _to_camel_key(key: str) -> str:
    if "_" in key:
        head, *tail = key.split("_")
        camel = head + "".join(part.capitalize() for part in tail)
    else:
        camel = key

    if camel.endswith("Id"):
        return camel[:-2] + "ID"
    return camel


def _to_camel_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {_to_camel_key(str(k)): _to_camel_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_camel_payload(item) for item in value]
    return value


@router.post(
    "/agent/chat",
    summary="Conversar com profile agent",
    description="Recebe uma mensagem do usuário e continua a conversa de criação de profile para o friendID.",
    responses={
        200: {"description": "Mensagem do agente retornada com sucesso"},
        400: {"description": "Payload inválido"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["friendID", "message"],
                        "properties": {
                            "friendID": {"type": "string", "format": "uuid"},
                            "message": {"type": "string"},
                        },
                    },
                    "example": {
                        "friendID": "c096d057-a0cf-4c2d-857b-41beccb42de8",
                        "message": "Ele ama jogos de tabuleiro e não curte lugares muito cheios",
                    },
                }
            },
        }
    },
)
async def chat_profile_agent(request: Request) -> dict[str, Any]:
    payload = await request.json()
    friend_id = _read_text(payload, "friendID", "friendId", "friend_id")
    message = str(payload.get("message", "")).strip()
    session_id = friend_id

    missing_fields = [
        field
        for field, value in {
            "friendID": friend_id,
            "message": message,
        }.items()
        if not value
    ]
    if missing_fields:
        raise HTTPException(status_code=400, detail=f"Campos obrigatorios ausentes: {', '.join(missing_fields)}")

    result = await _profile_agent_service.chat(session_id=session_id, friend_id=friend_id, user_message=message)
    return _to_camel_payload(result)


@router.post(
    "/agent/finalize",
    summary="Finalizar criação conversacional de profile",
    description="Consolida histórico da conversa, extrai profile, salva profile e embedding.",
    responses={
        200: {"description": "Profile finalizado com sucesso"},
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
                        "required": ["friendID"],
                        "properties": {
                            "friendID": {"type": "string", "format": "uuid"},
                        },
                    },
                    "example": {
                        "friendID": "c096d057-a0cf-4c2d-857b-41beccb42de8",
                    },
                }
            },
        }
    },
)
async def finalize_profile_agent(request: Request) -> dict[str, Any]:
    payload = await request.json()
    session_id = _read_text(payload, "friendID", "friendId", "friend_id", "sessionID", "sessionId", "session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Campo obrigatorio ausente: friendID")

    try:
        result = await _profile_agent_service.finalize_profile(session_id=session_id)
        return _to_camel_payload(result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete(
    "/agent/session/{sessionId}",
    summary="Limpar sessão do profile agent",
    description="Remove mensagens da sessão de conversa do profile agent pelo identificador (friendID).",
)
async def clear_profile_agent_session(sessionId: str) -> dict[str, Any]:
    return _to_camel_payload(_profile_agent_service.clear_session(sessionId))
