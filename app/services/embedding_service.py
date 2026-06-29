from typing import Any

from app.clients.data_backend_client import DataBackendClient
from app.clients.embedding_client import EmbeddingClient, build_profile_embedding_text


class EmbeddingService:
    def __init__(self, data_backend_client: DataBackendClient, embedding_client: EmbeddingClient) -> None:
        self._data_backend_client = data_backend_client
        self._embedding_client = embedding_client

    async def generate_and_save_profile_embedding(
        self,
        friend_id: str,
        profile_payload: dict[str, Any],
    ) -> dict[str, Any]:
        text = build_profile_embedding_text(profile_payload)
        embedding = await self._embedding_client.embed_text(text)
        request_payload: dict[str, Any] = {
            "friend_id": friend_id,
            "embedding": embedding,
        }

        if "likes" in profile_payload:
            request_payload["likes"] = profile_payload.get("likes")
        if "dislikes" in profile_payload:
            request_payload["dislikes"] = profile_payload.get("dislikes")

        return await self._data_backend_client.save_profile_embedding(
            friend_id,
            request_payload,
        )
