from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from app.core.settings import settings


def build_profile_agent():
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Voce e um assistente de coleta de perfil para presentes. Conduza uma conversa natural em portugues, com perguntas curtas e objetivas.\n"
                "Objetivo: descobrir likes e dislikes da pessoa alvo (friend_id).\n"
                "Regras:\n"
                "1) Pergunte uma coisa por vez.\n"
                "2) Nao invente informacoes.\n"
                "3) Se o usuario nao souber, marque como desconhecido e siga.\n"
                "4) Foque em detalhes que ajudam recomendacao: hobbies, estilo, preferencias e aversoes.\n"
                "5) Quando houver contexto suficiente, sinalize que esta pronto para finalizar com uma confirmacao curta.",
            ),
            ("system", "friend_id alvo da conversa: {friend_id}"),
            ("system", "Contexto conhecido da pessoa alvo: {friend_context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    return prompt | llm


def build_chat_history(messages: list[dict[str, str]]) -> list[HumanMessage | AIMessage]:
    history: list[HumanMessage | AIMessage] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "assistant":
            history.append(AIMessage(content=content))
        else:
            history.append(HumanMessage(content=content))
    return history
