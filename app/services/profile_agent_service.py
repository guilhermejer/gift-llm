import json
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.profile_agent import build_chat_history, build_profile_agent
from app.clients.data_backend_client import DataBackendRequestError
from app.core.settings import settings
from app.services.embedding_service import EmbeddingService
from app.services.profile_memory_service import ProfileMemoryService
from app.services.profile_service import ProfileService


class ProfileAgentService:
    def __init__(
        self,
        profile_service: ProfileService,
        embedding_service: EmbeddingService,
        memory_service: ProfileMemoryService,
    ) -> None:
        self._profile_service = profile_service
        self._embedding_service = embedding_service
        self._memory_service = memory_service
        self._agent_executor = build_profile_agent()

    async def chat(self, session_id: str, friend_id: str, user_message: str) -> dict[str, Any]:
        effective_session_id = friend_id
        state = self._memory_service.append_user_message(effective_session_id, friend_id, user_message)
        chat_history = build_chat_history(state.messages[:-1])
        friend_context = await self._build_friend_context(friend_id)

        result = await self._agent_executor.ainvoke(
            {
                "input": user_message,
                "friend_id": friend_id,
                "friend_context": friend_context,
                "chat_history": chat_history,
            }
        )

        assistant_message = str(getattr(result, "content", "")).strip()
        if assistant_message:
            self._memory_service.append_assistant_message(effective_session_id, friend_id, assistant_message)

        snapshot = self._memory_service.session_snapshot(effective_session_id)
        return {
            "session": snapshot,
            "assistant_message": assistant_message,
        }

    async def finalize_profile(self, session_id: str) -> dict[str, Any]:
        state = self._memory_service.get_session(session_id)
        if state is None:
            raise ValueError("Sessao nao encontrada")

        conversation_text = self._memory_service.build_conversation_text(session_id)
        friend_data = await self._get_friend_data(state.friend_id)
        extracted = await self._extract_profile_fields(conversation_text, state.friend_id, friend_data)

        likes = extracted.get("likes", []) or []
        dislikes = extracted.get("dislikes", []) or []
        personality = extracted.get("personality", []) or []

        if not likes:
            likes = ["desconhecido"]
        if not dislikes:
            dislikes = ["desconhecido"]

        profile_payload: dict[str, Any] = {
            "friend_id": state.friend_id,
            "likes": likes,
            "dislikes": dislikes,
            "personality": personality,
            "name": extracted.get("name", ""),
            "city": extracted.get("city", ""),
            "user_relation": extracted.get("user_relation", ""),
            "conversation_history": conversation_text,
        }

        profile_result = await self._profile_service.create_profile(
            {
                "friend_id": state.friend_id,
                "likes": likes,
                "dislikes": dislikes,
                "personality": personality,
            }
        )

        embedding_result = await self._embedding_service.generate_and_save_profile_embedding(
            state.friend_id,
            profile_payload,
        )

        return {
            "session": self._memory_service.session_snapshot(session_id),
            "profile": profile_result,
            "embedding": embedding_result,
            "extracted_profile": extracted,
        }

    def clear_session(self, session_id: str) -> dict[str, Any]:
        removed = self._memory_service.clear_session(session_id)
        return {"session_id": session_id, "cleared": removed}

    async def _extract_profile_fields(
        self,
        conversation_text: str,
        friend_id: str,
        friend_data: dict[str, Any],
    ) -> dict[str, Any]:
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
        )

        system_prompt = (
            "Extraia um resumo estruturado de perfil a partir da conversa. "
            "Retorne APENAS JSON valido, sem markdown, no formato: "
            '{"friend_id":"...","likes":["..."],"dislikes":["..."],"personality":["..."],"name":"...","city":"...","user_relation":"..."}. '
            "O campo personality deve conter tracos de personalidade observados (ex: introvertido, aventureiro, pratico). "
            "Se algo nao estiver claro, use string vazia ou lista vazia."
        )

        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"friend_id: {friend_id}\n"
                        f"Dados conhecidos da tabela friends: {json.dumps(friend_data, ensure_ascii=False)}\n"
                        f"Conversa:\n{conversation_text}"
                    )
                ),
            ]
        )

        raw = str(response.content).strip()
        data = _safe_json_load(raw)
        if not isinstance(data, dict):
            return {"friend_id": friend_id, "likes": [], "dislikes": []}

        likes_raw = data.get("likes")
        dislikes_raw = data.get("dislikes")
        personality_raw = data.get("personality")
        likes = likes_raw if isinstance(likes_raw, list) else []
        dislikes = dislikes_raw if isinstance(dislikes_raw, list) else []
        personality = personality_raw if isinstance(personality_raw, list) else []

        return {
            "friend_id": friend_id,
            "likes": [str(item).strip() for item in likes if str(item).strip()],
            "dislikes": [str(item).strip() for item in dislikes if str(item).strip()],
            "personality": [str(item).strip() for item in personality if str(item).strip()],
            "name": str(data.get("name", "")).strip(),
            "city": str(data.get("city", "")).strip(),
            "user_relation": str(data.get("user_relation", "")).strip(),
        }

    async def _get_friend_data(self, friend_id: str) -> dict[str, Any]:
        try:
            payload = await self._profile_service.get_friend(friend_id)
        except DataBackendRequestError:
            return {"friend_id": friend_id}

        if isinstance(payload, dict):
            return payload
        return {"friend_id": friend_id}

    async def _build_friend_context(self, friend_id: str) -> str:
        friend_data = await self._get_friend_data(friend_id)

        name = str(friend_data.get("Name") or friend_data.get("name") or "").strip()
        gender = str(friend_data.get("Gender") or friend_data.get("gender") or "").strip()
        city = str(friend_data.get("City") or friend_data.get("city") or "").strip()
        birth_date = str(friend_data.get("BirthDate") or friend_data.get("birthDate") or "").strip()
        user_relation = str(friend_data.get("UserRelation") or friend_data.get("userRelation") or "").strip()

        details: list[str] = []
        if name:
            details.append(f"nome={name}")
        if gender:
            details.append(f"genero={gender}")
        if birth_date:
            details.append(f"data_nascimento={birth_date}")
        if city:
            details.append(f"cidade={city}")
        if user_relation:
            details.append(f"relacao={user_relation}")

        if not details:
            return "sem dados da tabela friends"

        return "; ".join(details)


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
