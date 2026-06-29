# Gift LLM API

API em FastAPI para orquestrar agentes de perfil e sugestao de presentes/passeios usando LangChain + OpenAI.

## Estrutura atual
- `app/main.py`: inicializacao da API e health check
- `app/api/`: rotas de perfil, eventos e sugestoes
- `app/services/`: regras de orquestracao
- `app/clients/`: cliente HTTP para backend de dados + cliente de embedding
- `app/tools/langchain_tools.py`: tools usadas pelos agentes
- `app/agents/`: agentes de perfil e sugestao
- `mcp_server/`: exposicao das tools via MCP

## Variaveis de ambiente
- `OPENAI_API_KEY` (obrigatoria)
- `OPENAI_MODEL` (opcional, default: `gpt-5.4-nano`)
- `OPENAI_EMBEDDING_MODEL` (opcional, default: `text-embedding-3-small`)
- `DATABASE_URL` (obrigatoria para memoria de conversa do profile agent)
- `DATA_BACKEND_BASE_URL` (opcional, default: `http://localhost:8080`)
- `APP_HOST` (opcional, default: `0.0.0.0`)
- `APP_PORT` (opcional, default: `8000`)
- `MAX_RESPONSE_LOG_CHARS` (opcional, default: `8000`; limite de caracteres do payload de resposta logado)

## Instalar dependencias
```bash
pip install -r requirements.txt
```

## Rodar API
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Rodar MCP Server
```bash
python -m mcp_server.server
```

## Endpoints base
- `GET /health`
- `POST /profiles/agent/chat` (conversa com memoria em cache)
- `POST /profiles/agent/finalize` (salva profile e embedding com base em toda conversa)
- `DELETE /profiles/agent/session/{session_id}` (limpa memoria de conversa usando friend_id)
- `POST /profiles/{friend_id}/suggestions` (body deve conter `occasion_details`; cria sugestoes iniciais e sessao por `gift_id`)
- `POST /suggestions/agent/chat`
- `POST /suggestions/agent/finalize`

## Observacoes de integracao com o swagger
- Profiles: integra com `PUT/GET /friends/{friend_id}/profile`
- Sugestoes: integra com `PUT/GET /friends/{friend_id}/gifts` e cria conversa por `gift_id`
- Sugestoes refinadas apos conversa sao persistidas via `POST /gifts/{gift_id}` e mantem `reminder_id` quando presente
- Eventos: integra com reminders via `PUT/GET /users/{user_id}/reminders` com filtro por `friend_id`
