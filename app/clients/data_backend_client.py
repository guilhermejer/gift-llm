from typing import Any

import httpx

from app.core.settings import settings


class DataBackendRequestError(Exception):
    def __init__(self, status_code: int, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.details = details or ""


class DataBackendClient:
    def __init__(self) -> None:
        self._base_url = settings.data_backend_base_url.rstrip("/")
        self._timeout = settings.request_timeout_seconds

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.request(method=method, url=url, json=payload)
        except httpx.TimeoutException as exc:
            raise DataBackendRequestError(
                status_code=504,
                message="Timeout ao chamar backend de dados",
            ) from exc
        except httpx.RequestError as exc:
            raise DataBackendRequestError(
                status_code=502,
                message="Falha de conectividade com backend de dados",
                details=str(exc),
            ) from exc

        if response.is_error:
            raise DataBackendRequestError(
                status_code=response.status_code,
                message="Backend de dados retornou erro",
                details=response.text[:2000],
            )

        return response.json() if response.content else {}

    async def create_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        friend_id = str(
            payload.get("friendID")
            or payload.get("friendId")
            or payload.get("friend_id")
            or ""
        ).strip()
        if not friend_id:
            raise ValueError("Campo obrigatorio ausente: friendID")
        merged_payload = {**payload, "friendID": friend_id}
        merged_payload.pop("friendId", None)
        merged_payload.pop("friend_id", None)
        return await self._request("PUT", f"/friends/{friend_id}/profile", merged_payload)

    async def get_profile(self, friend_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/friends/{friend_id}/profile")

    async def get_friend(self, friend_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/friends/{friend_id}")

    async def update_profile(self, friend_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        merged_payload = {**payload, "friendID": friend_id}
        merged_payload.pop("friendId", None)
        merged_payload.pop("friend_id", None)
        return await self._request("PUT", f"/friends/{friend_id}/profile", merged_payload)

    async def save_profile_embedding(self, friend_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        merged_payload = {**payload, "friendID": friend_id}
        merged_payload.pop("friendId", None)
        merged_payload.pop("friend_id", None)
        return await self._request("PUT", f"/friends/{friend_id}/profile", merged_payload)

    async def create_friend_gift(self, friend_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("PUT", f"/friends/{friend_id}/gifts", payload)

    async def update_gift(self, gift_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", f"/gifts/{gift_id}", payload)

    async def list_friend_gifts(self, friend_id: str) -> list[dict[str, Any]]:
        response = await self._request("GET", f"/friends/{friend_id}/gifts")
        if isinstance(response, list):
            return response
        items = response.get("items")
        return items if isinstance(items, list) else []

    async def create_user_reminder(self, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("PUT", f"/users/{user_id}/reminders", payload)

    async def list_user_reminders(self, user_id: str) -> list[dict[str, Any]]:
        response = await self._request("GET", f"/users/{user_id}/reminders")
        if isinstance(response, list):
            return response
        items = response.get("items")
        return items if isinstance(items, list) else []
