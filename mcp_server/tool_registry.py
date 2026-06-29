from collections.abc import Callable
from typing import Any

from app.tools.langchain_tools import (
    create_or_update_profile,
    generate_gift_and_activity_suggestions,
    generate_profile_embedding,
)


def get_registered_tools() -> dict[str, Callable]:
    async def mcp_create_or_update_profile(payload: dict[str, Any]) -> dict[str, Any]:
        return await create_or_update_profile.ainvoke({"payload": payload})

    async def mcp_generate_profile_embedding(
        friend_id: str,
        profile_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await generate_profile_embedding.ainvoke(
            {
                "friend_id": friend_id,
                "profile_payload": profile_payload,
            }
        )

    async def mcp_generate_gift_and_activity_suggestions(
        friend_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await generate_gift_and_activity_suggestions.ainvoke(
            {
                "friend_id": friend_id,
                "payload": payload,
            }
        )

    return {
        "create_or_update_profile": mcp_create_or_update_profile,
        "generate_profile_embedding": mcp_generate_profile_embedding,
        "generate_gift_and_activity_suggestions": mcp_generate_gift_and_activity_suggestions,
    }
