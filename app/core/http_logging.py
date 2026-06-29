from __future__ import annotations

import time
from datetime import datetime
from uuid import uuid4

from fastapi import Request

from app.core.logging import log_http_event


class RequestResponseLoggingMiddleware:
    def __init__(self, app, max_payload_log_chars: int = 8000) -> None:
        self.app = app
        self.max_payload_log_chars = max_payload_log_chars

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        request_id = f"req_{uuid4().int}"
        start_monotonic = time.perf_counter()
        started_at = datetime.now()

        method = request.method
        path = request.url.path
        remote_addr = _remote_addr(request)
        user_agent = request.headers.get("user-agent", "")

        # Disponibiliza request_id para handlers/exception handlers
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        log_http_event(
            {
                "level": "info",
                "date": started_at.strftime("%Y-%m-%d"),
                "time": started_at.strftime("%H:%M:%S"),
                "message": "request started",
                "request_id": request_id,
                "method": method,
                "path": path,
                "remote_addr": remote_addr,
                "user_agent": user_agent,
            }
        )

        status_code = 500
        response_headers: list[tuple[bytes, bytes]] = []
        response_body_chunks: list[bytes] = []

        async def send_wrapper(message):
            nonlocal status_code, response_headers
            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 500))
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode("utf-8")))
                message["headers"] = response_headers
            elif message["type"] == "http.response.body":
                body = message.get("body", b"") or b""
                if body:
                    response_body_chunks.append(body)
            await send(message)

        await self.app(scope, receive, send_wrapper)

        latency_ms = int((time.perf_counter() - start_monotonic) * 1000)
        response_bytes = sum(len(chunk) for chunk in response_body_chunks)
        payload = _extract_payload(response_headers, response_body_chunks, self.max_payload_log_chars)

        completed_at = datetime.now()
        log_http_event(
            {
                "level": "info",
                "date": completed_at.strftime("%Y-%m-%d"),
                "time": completed_at.strftime("%H:%M:%S"),
                "message": "request completed",
                "request_id": request_id,
                "method": method,
                "path": path,
                "status": status_code,
                "latency_ms": latency_ms,
                "response_bytes": response_bytes,
                "payload": payload,
            }
        )


def _extract_payload(
    response_headers: list[tuple[bytes, bytes]],
    response_body_chunks: list[bytes],
    max_payload_log_chars: int,
) -> str:
    if not response_body_chunks:
        return ""

    content_type = ""
    for key, value in response_headers:
        if key.lower() == b"content-type":
            content_type = value.decode("utf-8", errors="ignore")
            break

    is_text_payload = any(
        marker in content_type.lower()
        for marker in ("application/json", "text/", "application/problem+json")
    )
    if not is_text_payload:
        return "<non-textual-payload>"

    raw = b"".join(response_body_chunks)
    text = raw.decode("utf-8", errors="replace")
    if len(text) > max_payload_log_chars:
        return f"{text[:max_payload_log_chars]}...(truncated)"
    return text


def _remote_addr(request: Request) -> str:
    if request.client is None:
        return ""
    host = request.client.host or ""
    port = request.client.port
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{host}:{port}" if port is not None else host
