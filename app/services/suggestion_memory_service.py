from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import psycopg


@dataclass
class SuggestionConversationState:
    session_id: str
    gift_id: str
    friend_id: str
    occasion_context: str
    gift_context: str
    reminder_id: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SuggestionMemoryService:
    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL nao configurada")

        self._database_url = database_url
        self._lock = Lock()
        self._ensure_tables()

    def _connection(self):
        return psycopg.connect(self._database_url)

    def _ensure_tables(self) -> None:
        query = """
        CREATE TABLE IF NOT EXISTS suggestion_agent_sessions (
            session_id TEXT PRIMARY KEY,
            gift_id TEXT NOT NULL,
            friend_id TEXT NOT NULL,
            occasion_context TEXT NOT NULL,
            gift_context TEXT NOT NULL,
            reminder_id TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        ALTER TABLE suggestion_agent_sessions
        ADD COLUMN IF NOT EXISTS reminder_id TEXT NOT NULL DEFAULT '';

        CREATE TABLE IF NOT EXISTS suggestion_agent_messages (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_suggestion_agent_messages_session
            ON suggestion_agent_messages (session_id, id);
        """

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
            conn.commit()

    def initialize_session(
        self,
        gift_id: str,
        friend_id: str,
        occasion_context: str,
        gift_context: str,
        reminder_id: str = "",
    ) -> dict[str, Any]:
        normalized_session_id = gift_id
        with self._lock:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO suggestion_agent_sessions (
                            session_id,
                            gift_id,
                            friend_id,
                            occasion_context,
                            gift_context,
                            reminder_id
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (session_id)
                        DO UPDATE SET
                            friend_id = EXCLUDED.friend_id,
                            occasion_context = EXCLUDED.occasion_context,
                            gift_context = EXCLUDED.gift_context,
                            reminder_id = EXCLUDED.reminder_id,
                            updated_at = NOW()
                        """,
                        (normalized_session_id, gift_id, friend_id, occasion_context, gift_context, reminder_id),
                    )
                conn.commit()

        return self.session_snapshot(normalized_session_id)

    def _load_messages(self, session_id: str) -> list[dict[str, str]]:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content
                    FROM suggestion_agent_messages
                    WHERE session_id = %s
                    ORDER BY id ASC
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()

        return [{"role": str(row[0]), "content": str(row[1])} for row in rows]

    def get_session(self, session_id: str) -> SuggestionConversationState | None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT gift_id, friend_id, occasion_context, gift_context, reminder_id, updated_at
                    FROM suggestion_agent_sessions
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                row = cur.fetchone()

        if not row:
            return None

        updated_at = row[5] if isinstance(row[5], datetime) else datetime.now(timezone.utc)
        return SuggestionConversationState(
            session_id=session_id,
            gift_id=str(row[0]),
            friend_id=str(row[1]),
            occasion_context=str(row[2]),
            gift_context=str(row[3]),
            reminder_id=str(row[4]),
            messages=self._load_messages(session_id),
            updated_at=updated_at,
        )

    def _append_message(self, session_id: str, role: str, content: str) -> SuggestionConversationState:
        with self._lock:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO suggestion_agent_messages (session_id, role, content)
                        VALUES (%s, %s, %s)
                        """,
                        (session_id, role, content),
                    )
                    cur.execute(
                        "UPDATE suggestion_agent_sessions SET updated_at = NOW() WHERE session_id = %s",
                        (session_id,),
                    )
                conn.commit()

        state = self.get_session(session_id)
        if state is None:
            raise ValueError("Sessao de sugestao nao encontrada")
        return state

    def append_user_message(self, session_id: str, content: str) -> SuggestionConversationState:
        return self._append_message(session_id, "user", content)

    def append_assistant_message(self, session_id: str, content: str) -> SuggestionConversationState:
        return self._append_message(session_id, "assistant", content)

    def clear_session(self, session_id: str) -> bool:
        with self._lock:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM suggestion_agent_messages WHERE session_id = %s", (session_id,))
                    removed_messages = cur.rowcount > 0
                    cur.execute("DELETE FROM suggestion_agent_sessions WHERE session_id = %s", (session_id,))
                    removed_session = cur.rowcount > 0
                conn.commit()
        return removed_messages or removed_session

    def update_gift_context(self, session_id: str, gift_context: str) -> SuggestionConversationState:
        with self._lock:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE suggestion_agent_sessions SET gift_context = %s, updated_at = NOW() WHERE session_id = %s",
                        (gift_context, session_id),
                    )
                conn.commit()

        state = self.get_session(session_id)
        if state is None:
            raise ValueError("Sessao de sugestao nao encontrada")
        return state

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
        state = self.get_session(session_id)
        if state is None:
            return {"session_id": session_id, "exists": False}

        return {
            "session_id": state.session_id,
            "gift_id": state.gift_id,
            "friend_id": state.friend_id,
            "exists": True,
            "message_count": len(state.messages),
            "updated_at": state.updated_at.isoformat(),
            "occasion_context": state.occasion_context,
            "gift_context": state.gift_context,
            "reminder_id": state.reminder_id,
        }