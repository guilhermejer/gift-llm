from typing import Any

from app.clients.data_backend_client import DataBackendClient


class SuggestionService:
    def __init__(self, data_backend_client: DataBackendClient) -> None:
        self._data_backend_client = data_backend_client

    async def create_suggestions(self, friend_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        suggestions = payload.get("suggestions", [])
        reminder_id = str(
            payload.get("reminderID")
            or payload.get("reminderId")
            or payload.get("reminder_id")
            or ""
        ).strip()
        saved_items: list[dict[str, Any]] = []

        for suggestion in suggestions:
            gift_payload = {
                "friendID": friend_id,
                "title": suggestion.get("title", ""),
                "description": suggestion.get("reason") or suggestion.get("description", ""),
                "priceRange": suggestion.get("priceRange") or suggestion.get("price_range", ""),
                "type": suggestion.get("type", "gift") or "gift",
            }
            tags_raw = suggestion.get("tags", [])
            if isinstance(tags_raw, list) and tags_raw:
                gift_payload["tags"] = tags_raw
            if reminder_id:
                gift_payload["reminderID"] = reminder_id
            saved_items.append(await self._data_backend_client.create_friend_gift(friend_id, gift_payload))

        return {
            "friendID": friend_id,
            "saved": len(saved_items),
            "items": saved_items,
        }

    async def update_suggestion(self, gift_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._data_backend_client.update_gift(gift_id, payload)
