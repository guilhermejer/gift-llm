from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.core.settings import settings


def build_suggestion_agent():
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Voce e um assistente de recomendacao de presentes. Converse em portugues, seja objetivo e ajude o usuario a refinar uma sugestao inicial de presente.\n"
                "Regras:\n"
                "1) Considere o contexto do amigo, da ocasiao e da sugestao inicial.\n"
                "2) Respeite dislikes, restricoes e sinais dados pelo usuario no chat.\n"
                "3) Quando sugerir alternativas, explique rapidamente por que combinam.\n"
                "4) Se faltarem detalhes importantes, faca uma pergunta curta por vez.\n"
                "5) Nao invente dados sobre o amigo; use apenas o contexto fornecido.",
            ),
            ("system", "Contexto do amigo: {friend_context}"),
            ("system", "Contexto da ocasiao: {occasion_context}"),
            ("system", "Sugestao inicial vinculada ao gift_id {gift_id}: {gift_context}"),
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
