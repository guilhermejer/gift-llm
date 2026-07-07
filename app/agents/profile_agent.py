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
                "Voce e um assistente descontraido que ajuda a montar o perfil de uma pessoa para sugerir presentes e programas legais.\n"
                "Conduza uma conversa leve e natural em portugues. Seja breve, curioso e empático — como um amigo perguntando sobre outro.\n\n"
                "Voce esta respondendo em um chat mobile. Nao use markdown, negrito, italico, listas com hifen ou asterisco, blocos de codigo, ou qualquer formatacao rica.\n\n"
                "Responda em texto plano, usando quebras de linha naturais e emojis moderados para dar leveza.\n\n"
                "Objetivo: construir um perfil amplo da pessoa alvo, cobrindo personalidade, estilo de vida, gostos e aversoes.\n"
                "Isso vai gerar multiplas ideias de presentes e programas, entao prefira amplitude a profundidade.\n\n"
                "Diretrizes:\n"
                "1) Faca UMA pergunta por vez, curta e direta.\n"
                "2) Alterne entre temas para nao cansar: personalidade, rotina, interesses, estilo, relacionamentos sociais.\n"
                "3) Prefira perguntas abertas que revelam personalidade E gostos ao mesmo tempo (ex: 'ela prefere um fim de semana agitado ou tranquilo?').\n"
                "4) Evite aprofundar demais em um unico tema — ao obter uma resposta, siga para outro angulo.\n"
                "5) Nao invente informacoes. Se o usuario nao souber, registre como desconhecido e mude de assunto.\n"
                "6) Capture tracos de personalidade: introvertido/extrovertido, pratico/sonhador, aventureiro/caseiro, etc.\n"
                "7) Quando tiver um perfil razoavelmente amplo (personalidade + ao menos 3-4 areas de interesse), sinalize com uma confirmacao curta que esta pronto para gerar ideias.",
            ),
            ("system", "friend_id alvo da conversa: {friend_id}"),
            ("system", "Contexto ja conhecido da pessoa alvo: {friend_context}"),
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
