import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.suggestion_agent import build_chat_history, build_suggestion_agent
from app.clients.data_backend_client import DataBackendRequestError
from app.core.settings import settings
from app.services.profile_service import ProfileService
from app.services.suggestion_memory_service import SuggestionMemoryService
from app.services.suggestion_service import SuggestionService


class SuggestionAgentService:
    def __init__(
        self,
        suggestion_service: SuggestionService,
        profile_service: ProfileService,
        memory_service: SuggestionMemoryService,
    ) -> None:
        self._suggestion_service = suggestion_service
        self._profile_service = profile_service
        self._memory_service = memory_service
        self._agent_executor = build_suggestion_agent()

    async def create_initial_suggestions(self, friend_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        occasion_context = _build_occasion_context(payload)
        reminder_id = _read_text(payload, "reminder_id", "reminderID", "ReminderID")

        raw_suggestions = payload.get("suggestions")
        if isinstance(raw_suggestions, list) and raw_suggestions:
            suggestions = raw_suggestions
        else:
            suggestions = await self._generate_initial_suggestions(friend_id, occasion_context, payload)

        saved_result = await self._suggestion_service.create_suggestions(
            friend_id,
            {"suggestions": suggestions, "reminder_id": reminder_id},
        )

        sessions: list[dict[str, Any]] = []
        for saved_item, suggestion in zip(saved_result.get("items", []), suggestions):
            gift_id = _extract_gift_id(saved_item)
            if not gift_id:
                continue

            gift_context = _build_gift_context(saved_item, suggestion)
            snapshot = self._memory_service.initialize_session(
                gift_id=gift_id,
                friend_id=friend_id,
                occasion_context=occasion_context,
                gift_context=gift_context,
                reminder_id=_extract_reminder_id(saved_item) or reminder_id,
            )
            sessions.append(snapshot)

        return {
            **saved_result,
            "occasion_context": occasion_context,
            "reminder_id": reminder_id,
            "sessions": sessions,
        }

    async def chat(
        self,
        gift_id: str,
        user_message: str,
        friend_id: str | None = None,
        occasion_details: str = "",
    ) -> dict[str, Any]:
        state = self._memory_service.get_session(gift_id)
        if state is None:
            normalized_friend_id = str(friend_id or "").strip()
            if not normalized_friend_id:
                raise ValueError("Sessao de sugestao nao encontrada para esse gift_id")

            occasion_context = occasion_details.strip() or "ocasiao nao informada"
            snapshot = self._memory_service.initialize_session(
                gift_id=gift_id,
                friend_id=normalized_friend_id,
                occasion_context=occasion_context,
                gift_context="sugestao inicial nao encontrada",
                reminder_id="",
            )
            state = self._memory_service.get_session(snapshot["session_id"])

        if state is None:
            raise ValueError("Sessao de sugestao nao encontrada para esse gift_id")

        state = self._memory_service.append_user_message(state.session_id, user_message)
        chat_history = build_chat_history(state.messages[:-1])
        friend_context = await self._build_friend_context(state.friend_id)

        result = await self._agent_executor.ainvoke(
            {
                "input": user_message,
                "gift_id": state.gift_id,
                "friend_context": friend_context,
                "occasion_context": state.occasion_context,
                "gift_context": state.gift_context,
                "chat_history": chat_history,
            }
        )

        assistant_message = str(getattr(result, "content", "")).strip()
        if assistant_message:
            self._memory_service.append_assistant_message(state.session_id, assistant_message)

        return {
            "session": self._memory_service.session_snapshot(state.session_id),
            "assistant_message": assistant_message,
        }

    async def finalize_suggestion(self, gift_id: str, friend_id: str | None = None) -> dict[str, Any]:
        state = self._memory_service.get_session(gift_id)
        if state is None:
            raise ValueError("Sessao de sugestao nao encontrada")

        if friend_id and state.friend_id and friend_id != state.friend_id:
            raise ValueError("gift_id nao pertence ao friend_id informado")

        conversation_text = self._memory_service.build_conversation_text(gift_id)
        friend_context = await self._build_friend_context(state.friend_id)
        extracted = await self._extract_suggestion_fields(
            gift_id=state.gift_id,
            friend_id=state.friend_id,
            friend_context=friend_context,
            occasion_context=state.occasion_context,
            gift_context=state.gift_context,
            reminder_id=state.reminder_id,
            conversation_text=conversation_text,
        )

        update_payload = {
            "giftID": state.gift_id,
            "friendID": state.friend_id,
            "title": extracted.get("title", ""),
            "description": extracted.get("description", ""),
            "priceRange": extracted.get("price_range", ""),
            "tags": extracted.get("tags", ["gift"]),
        }
        if state.reminder_id:
            update_payload["reminderID"] = state.reminder_id

        persisted = await self._suggestion_service.update_suggestion(state.gift_id, update_payload)

        persisted_context = _build_gift_context(
            persisted if isinstance(persisted, dict) else update_payload,
            {
                "title": extracted.get("title", ""),
                "reason": extracted.get("description", ""),
                "price_range": extracted.get("price_range", ""),
            },
        )
        self._memory_service.update_gift_context(gift_id, persisted_context)

        return {
            "session": self._memory_service.session_snapshot(gift_id),
            "gift": persisted,
            "extracted_suggestion": extracted,
        }

    def clear_session(self, gift_id: str) -> dict[str, Any]:
        removed = self._memory_service.clear_session(gift_id)
        return {"session_id": gift_id, "cleared": removed}

    async def _generate_initial_suggestions(
        self,
        friend_id: str,
        occasion_context: str,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        friend_context = await self._build_friend_context(friend_id)
        suggestion_count = _safe_positive_int(payload.get("suggestion_count"), default=3)

        prompt = (
            "Gere sugestoes iniciais de presentes em portugues com base no contexto abaixo. "
            "Retorne APENAS JSON valido, sem markdown, no formato: "
            '{"suggestions":[{"title":"...","reason":"...","price_range":"...","type":"gift"}]}. '
            f"Crie exatamente {suggestion_count} sugestoes objetivas e diferentes entre si."
        )

        result = await self._agent_executor.ainvoke(
            {
                "input": prompt,
                "gift_id": "batch-initial",
                "friend_context": friend_context,
                "occasion_context": occasion_context,
                "gift_context": "gerar sugestoes iniciais para iniciar conversas futuras",
                "chat_history": [],
            }
        )

        parsed = _safe_json_load(str(getattr(result, "content", "")))
        suggestions = parsed.get("suggestions") if isinstance(parsed, dict) else None
        if not isinstance(suggestions, list):
            raise ValueError("Nao foi possivel gerar sugestoes iniciais validas")

        normalized: list[dict[str, Any]] = []
        for item in suggestions:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not title or not reason:
                continue
            normalized.append(
                {
                    "title": title,
                    "reason": reason,
                    "price_range": str(item.get("price_range", "")).strip(),
                    "type": str(item.get("type", "gift")).strip() or "gift",
                }
            )

        if not normalized:
            raise ValueError("Nao foi possivel gerar sugestoes iniciais validas")

        return normalized[:suggestion_count]

    async def _build_friend_context(self, friend_id: str) -> str:
        friend_payload = await self._get_friend_payload(friend_id)
        profile_payload = await self._get_profile_payload(friend_id)

        details: list[str] = []

        name = _read_value(friend_payload, "Name", "name")
        gender = _read_value(friend_payload, "Gender", "gender")
        city = _read_value(friend_payload, "City", "city")
        birth_date = _read_value(friend_payload, "BirthDate", "birthDate")
        relation = _read_value(friend_payload, "UserRelation", "userRelation")
        likes = _read_value(profile_payload, "likes", "Likes")
        dislikes = _read_value(profile_payload, "dislikes", "Dislikes")

        if name:
            details.append(f"nome={name}")
        if gender:
            details.append(f"genero={gender}")
        if city:
            details.append(f"cidade={city}")
        if birth_date:
            details.append(f"data_nascimento={birth_date}")
        if relation:
            details.append(f"relacao={relation}")
        personality = _read_value(profile_payload, "personality", "Personality")
        if isinstance(likes, list) and likes:
            details.append("likes=" + ", ".join(str(item).strip() for item in likes if str(item).strip()))
        if isinstance(dislikes, list) and dislikes:
            details.append("dislikes=" + ", ".join(str(item).strip() for item in dislikes if str(item).strip()))
        if isinstance(personality, list) and personality:
            details.append("personalidade=" + ", ".join(str(item).strip() for item in personality if str(item).strip()))
        elif isinstance(personality, str) and personality:
            details.append(f"personalidade={personality}")
        if "embedding" in profile_payload or "Embedding" in profile_payload:
            details.append("embedding_de_profile_persistido=disponivel")

        if not details:
            return "sem contexto persistido do friend"

        return "; ".join(details)

    async def _get_friend_payload(self, friend_id: str) -> dict[str, Any]:
        try:
            payload = await self._profile_service.get_friend(friend_id)
        except DataBackendRequestError:
            return {}
        return payload if isinstance(payload, dict) else {}

    async def _get_profile_payload(self, friend_id: str) -> dict[str, Any]:
        try:
            payload = await self._profile_service.get_profile(friend_id)
        except DataBackendRequestError:
            return {}
        return payload if isinstance(payload, dict) else {}

    async def _extract_suggestion_fields(
        self,
        gift_id: str,
        friend_id: str,
        friend_context: str,
        occasion_context: str,
        gift_context: str,
        reminder_id: str,
        conversation_text: str,
    ) -> dict[str, Any]:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
        )

        system_prompt = (
            "Extraia a versao final da sugestao de presente a partir do historico da conversa. "
            "Retorne APENAS JSON valido, sem markdown, no formato: "
            '{"gift_id":"...","friend_id":"...","title":"...","description":"...","price_range":"...","tags":["gift"],"reminder_id":"..."}. '
            "Se algum campo nao estiver claro, preserve a melhor opcao disponivel no contexto atual."
        )

        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"gift_id: {gift_id}\n"
                        f"friend_id: {friend_id}\n"
                        f"reminder_id: {reminder_id}\n"
                        f"contexto_do_amigo: {friend_context}\n"
                        f"contexto_da_ocasiao: {occasion_context}\n"
                        f"sugestao_atual: {gift_context}\n"
                        f"conversa:\n{conversation_text}"
                    )
                ),
            ]
        )

        raw = str(response.content).strip()
        data = _safe_json_load(raw)
        if not isinstance(data, dict):
            return _fallback_suggestion_payload(gift_id, friend_id, gift_context, reminder_id)

        tags_raw = data.get("tags")
        tags = [str(item).strip() for item in tags_raw if str(item).strip()] if isinstance(tags_raw, list) else []
        if not tags:
            tags = ["gift"]

        fallback = _fallback_suggestion_payload(gift_id, friend_id, gift_context, reminder_id)
        return {
            "gift_id": gift_id,
            "friend_id": friend_id,
            "title": _coalesce_text(data.get("title"), fallback["title"]),
            "description": _coalesce_text(data.get("description"), fallback["description"]),
            "price_range": _coalesce_text(data.get("price_range"), fallback["price_range"]),
            "tags": tags,
            "reminder_id": _coalesce_text(data.get("reminder_id"), reminder_id),
        }


def _extract_gift_id(payload: dict[str, Any]) -> str:
    for key in ("giftID", "GiftID", "giftId", "gift_id", "id", "ID"):
        value = payload.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _extract_reminder_id(payload: dict[str, Any]) -> str:
    return _read_text(payload, "reminderID", "ReminderID", "reminder_id", "reminderId")


def _build_gift_context(saved_item: dict[str, Any], suggestion: dict[str, Any]) -> str:
    title = _read_value(saved_item, "title", "Title") or str(suggestion.get("title", "")).strip()
    description = _read_value(saved_item, "description", "Description") or str(suggestion.get("reason", "")).strip()
    price_range = _read_value(saved_item, "priceRange", "price_range", "PriceRange") or str(
        suggestion.get("price_range", "")
    ).strip()
    reminder_id = _read_text(saved_item, "reminderID", "ReminderID", "reminder_id", "reminderId")

    parts = [
        f"titulo={title}" if title else "",
        f"motivo={description}" if description else "",
        f"faixa_preco={price_range}" if price_range else "",
        f"reminder_id={reminder_id}" if reminder_id else "",
    ]
    return "; ".join([part for part in parts if part]) or "sugestao inicial sem detalhes"


def _build_occasion_context(payload: dict[str, Any]) -> str:
    source = str(payload.get("source", "on_demand")).strip() or "on_demand"
    occasion_type = str(payload.get("occasion_type", "")).strip()
    occasion_name = str(payload.get("occasion_name", payload.get("event_name", ""))).strip()
    occasion_date = str(payload.get("occasion_date", payload.get("event_date", ""))).strip()
    occasion_details = str(payload.get("occasion_details", "")).strip()

    details: list[str] = [f"origem={source}"]
    if occasion_type:
        details.append(f"tipo={occasion_type}")
    if occasion_name:
        details.append(f"nome={occasion_name}")
    if occasion_date:
        details.append(f"data={occasion_date}")
    if occasion_details:
        details.append(f"detalhes={occasion_details}")

    return "; ".join(details)


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _read_value(payload: dict[str, Any], *keys: str) -> str | list[Any]:
    for key in keys:
        if key in payload:
            value = payload[key]
            if isinstance(value, list):
                return value
            return str(value).strip()
    return ""


def _read_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in payload and str(payload[key] or "").strip():
            return str(payload[key]).strip()
    return ""


def _coalesce_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _fallback_suggestion_payload(
    gift_id: str,
    friend_id: str,
    gift_context: str,
    reminder_id: str,
) -> dict[str, Any]:
    title = _extract_context_value(gift_context, "titulo") or "Sugestao refinada"
    description = _extract_context_value(gift_context, "motivo") or "Sugestao refinada a partir da conversa"
    price_range = _extract_context_value(gift_context, "faixa_preco") or "a definir"
    return {
        "gift_id": gift_id,
        "friend_id": friend_id,
        "title": title,
        "description": description,
        "price_range": price_range,
        "tags": ["gift"],
        "reminder_id": reminder_id,
    }


def _extract_context_value(text: str, key: str) -> str:
    for part in text.split(";"):
        normalized = part.strip()
        prefix = f"{key}="
        if normalized.startswith(prefix):
            return normalized[len(prefix) :].strip()
    return ""


def _safe_json_load(text: str) -> Any:
    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        snippet = text[first : last + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return {}

    return {}