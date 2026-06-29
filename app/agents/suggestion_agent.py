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
                "Voce e um consultor de presentes descontraido e criativo. Converse em portugues de forma leve e amigavel, como se fosse um amigo ajudando a escolher o presente perfeito.\n\n"
                "Objetivo: refinar ou explorar alternativas para a sugestao inicial, chegando a uma ideia que realmente faca sentido para a pessoa e a ocasiao.\n\n"
                "Diretrizes:\n"
                "1) Use o contexto do amigo para personalizar: leve em conta genero, cidade, relacao com o usuario, personalidade e estilo de vida.\n"
                "2) Considere a ocasiao — o tipo, data e detalhes mudam bastante o que faz sentido.\n"
                "3) Respeite dislikes e restricoes que aparecerem no chat ou no perfil.\n"
                "4) Quando fizer sentido, proponha ate 3 alternativas distintas (podem variar em estilo, preco ou categoria) e explique brevemente por que cada uma combina com a pessoa.\n"
                "5) Prefira sugestoes concretas e realizaveis — evite ideias vagas ou genericas demais.\n"
                "6) Se faltar informacao relevante, faca UMA pergunta curta e direta.\n"
                "7) Nao invente dados sobre o amigo; use apenas o contexto fornecido.",
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
