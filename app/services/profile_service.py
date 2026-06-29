from typing import Any

from app.clients.data_backend_client import DataBackendClient


class ProfileService:
    def __init__(self, data_backend_client: DataBackendClient) -> None:
        self._data_backend_client = data_backend_client

    async def create_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._data_backend_client.create_profile(payload)

    async def get_profile(self, friend_id: str) -> dict[str, Any]:
        return await self._data_backend_client.get_profile(friend_id)

    async def get_friend(self, friend_id: str) -> dict[str, Any]:
        return await self._data_backend_client.get_friend(friend_id)

    async def update_profile(self, friend_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._data_backend_client.update_profile(friend_id, payload)
