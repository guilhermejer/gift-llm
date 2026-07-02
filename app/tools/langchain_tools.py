from typing import Any

from langchain_core.tools import tool

from app.clients.data_backend_client import DataBackendClient
from app.clients.embedding_client import EmbeddingClient
from app.services.embedding_service import EmbeddingService
from app.services.profile_service import ProfileService
from app.services.suggestion_service import SuggestionService

_data_backend_client = DataBackendClient()
_profile_service = ProfileService(_data_backend_client)
_suggestion_service = SuggestionService(_data_backend_client)
_embedding_service = EmbeddingService(_data_backend_client, EmbeddingClient())


@tool
async def create_or_update_profile(payload: dict[str, Any]) -> dict[str, Any]:
    """Cria ou atualiza perfil no backend de dados."""
    friend_id = payload.get("friendID") or payload.get("friendId") or payload.get("friend_id")
    if friend_id:
        return await _profile_service.update_profile(str(friend_id), payload)
    return await _profile_service.create_profile(payload)


@tool
async def generate_profile_embedding(friend_id: str, profile_payload: dict[str, Any]) -> dict[str, Any]:
    """Gera embedding do perfil e envia para persistencia no backend."""
    return await _embedding_service.generate_and_save_profile_embedding(friend_id, profile_payload)


@tool
async def generate_gift_and_activity_suggestions(friend_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Persiste sugestoes de presentes e passeios no backend de dados."""
    return await _suggestion_service.create_suggestions(friend_id, payload)
