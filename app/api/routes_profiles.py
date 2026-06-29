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


@router.post(
    "/agent/chat",
    summary="Conversar com profile agent",
    description="Recebe uma mensagem do usuário e continua a conversa de criação de profile para o friend_id.",
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
                        "required": ["friend_id", "message"],
                        "properties": {
                            "friend_id": {"type": "string", "format": "uuid"},
                            "message": {"type": "string"},
                        },
                    },
                    "example": {
                        "friend_id": "c096d057-a0cf-4c2d-857b-41beccb42de8",
                        "message": "Ele ama jogos de tabuleiro e não curte lugares muito cheios",
                    },
                }
            },
        }
    },
)
async def chat_profile_agent(request: Request) -> dict[str, Any]:
    payload = await request.json()
    friend_id = str(payload.get("friend_id", "")).strip()
    message = str(payload.get("message", "")).strip()
    session_id = friend_id

    missing_fields = [
        field
        for field, value in {
            "friend_id": friend_id,
            "message": message,
        }.items()
        if not value
    ]
    if missing_fields:
        raise HTTPException(status_code=400, detail=f"Campos obrigatorios ausentes: {', '.join(missing_fields)}")

    return await _profile_agent_service.chat(session_id=session_id, friend_id=friend_id, user_message=message)


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
                        "required": ["friend_id"],
                        "properties": {
                            "friend_id": {"type": "string", "format": "uuid"},
                        },
                    },
                    "example": {
                        "friend_id": "c096d057-a0cf-4c2d-857b-41beccb42de8",
                    },
                }
            },
        }
    },
)
async def finalize_profile_agent(request: Request) -> dict[str, Any]:
    payload = await request.json()
    session_id = str(payload.get("friend_id", "")).strip() or str(payload.get("session_id", "")).strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="Campo obrigatorio ausente: friend_id")

    try:
        return await _profile_agent_service.finalize_profile(session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete(
    "/agent/session/{session_id}",
    summary="Limpar sessão do profile agent",
    description="Remove mensagens da sessão de conversa do profile agent pelo identificador (friend_id).",
)
async def clear_profile_agent_session(session_id: str) -> dict[str, Any]:
    return _profile_agent_service.clear_session(session_id)
