from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import psycopg


@dataclass
class ProfileConversationState:
    session_id: str
    friend_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ProfileMemoryService:
    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL nao configurada")

        self._database_url = database_url
        self._lock = Lock()
        self._ensure_table()

    def _connection(self):
        return psycopg.connect(self._database_url)

    def _ensure_table(self) -> None:
        query = """
        CREATE TABLE IF NOT EXISTS profile_agent_memory_messages (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            friend_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_profile_agent_memory_session
            ON profile_agent_memory_messages (session_id, id);
        """

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
            conn.commit()

    def _load_messages(self, session_id: str) -> list[dict[str, str]]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content
                    FROM profile_agent_memory_messages
                    WHERE session_id = %s
                    ORDER BY id ASC
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()

        return [{"role": str(row[0]), "content": str(row[1])} for row in rows]

    def _last_updated_at(self, session_id: str) -> datetime:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT MAX(created_at)
                    FROM profile_agent_memory_messages
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                row = cur.fetchone()

        value = row[0] if row else None
        if isinstance(value, datetime):
            return value
        return datetime.now(timezone.utc)

    def get_or_create_session(self, session_id: str, friend_id: str) -> ProfileConversationState:
        normalized_session_id = friend_id
        messages = self._load_messages(normalized_session_id)
        return ProfileConversationState(
            session_id=normalized_session_id,
            friend_id=friend_id,
            messages=messages,
            updated_at=self._last_updated_at(normalized_session_id),
        )

    def _append_message(self, session_id: str, friend_id: str, role: str, content: str) -> ProfileConversationState:
        normalized_session_id = friend_id
        with self._lock:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO profile_agent_memory_messages (session_id, friend_id, role, content)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (normalized_session_id, friend_id, role, content),
                    )
                conn.commit()

        return self.get_or_create_session(normalized_session_id, friend_id)

    def append_user_message(self, session_id: str, friend_id: str, content: str) -> ProfileConversationState:
        return self._append_message(session_id, friend_id, "user", content)

    def append_assistant_message(self, session_id: str, friend_id: str, content: str) -> ProfileConversationState:
        return self._append_message(session_id, friend_id, "assistant", content)

    def get_session(self, session_id: str) -> ProfileConversationState | None:
        normalized_session_id = session_id
        messages = self._load_messages(normalized_session_id)
        if not messages:
            return None

        friend_id = normalized_session_id
        return ProfileConversationState(
            session_id=normalized_session_id,
            friend_id=friend_id,
            messages=messages,
            updated_at=self._last_updated_at(normalized_session_id),
        )

    def clear_session(self, session_id: str) -> bool:
        with self._lock:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM profile_agent_memory_messages WHERE session_id = %s",
                        (session_id,),
                    )
                    removed = cur.rowcount > 0
                conn.commit()
        return removed

    def build_conversation_text(self, session_id: str) -> str:
        state = self.get_session(session_id)
        if state is None:
            return ""
        lines: list[str] = []
        for msg in state.messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def session_snapshot(self, session_id: str) -> dict[str, Any]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*), MAX(created_at)
                    FROM profile_agent_memory_messages
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                row = cur.fetchone()

        count = int(row[0]) if row and row[0] is not None else 0
        if count == 0:
            return {"session_id": session_id, "exists": False}

        updated_at = row[1] if row and isinstance(row[1], datetime) else datetime.now(timezone.utc)
        return {
            "session_id": session_id,
            "friend_id": session_id,
            "exists": True,
            "message_count": count,
            "updated_at": updated_at.isoformat(),
        }
