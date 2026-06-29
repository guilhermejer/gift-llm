from typing import Any

from langchain_openai import OpenAIEmbeddings

from app.core.settings import settings


class EmbeddingClient:
    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=lambda: settings.openai_api_key,
        )

    async def embed_text(self, text: str) -> list[float]:
        vectors: list[list[float]] = await self._embeddings.aembed_documents([text])
        if not vectors:
            return []
        return vectors[0]


def build_profile_embedding_text(profile_payload: dict[str, Any]) -> str:
    fields = [
        str(profile_payload.get("friend_id", "")),
        ", ".join(profile_payload.get("likes", []) or []),
        ", ".join(profile_payload.get("dislikes", []) or []),
        str(profile_payload.get("name", "")),
        str(profile_payload.get("city", "")),
        str(profile_payload.get("user_relation", "")),
        str(profile_payload.get("conversation_history", "")),
    ]
    return " | ".join([field for field in fields if field])
