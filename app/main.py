import logging

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from app.api.routes_profiles import router as profiles_router
from app.api.routes_suggestions import router as suggestions_router
from app.clients.data_backend_client import DataBackendRequestError
from app.core.http_logging import RequestResponseLoggingMiddleware
from app.core.logging import setup_logging
from app.core.settings import settings

setup_logging()
_logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "API de orquestração para perfil conversacional e sugestões de presentes/passeios. "
        "Integra com backend de dados (Go) e OpenAI."
    ),
)
app.add_middleware(
    RequestResponseLoggingMiddleware,
    max_payload_log_chars=settings.max_response_log_chars,
)


@app.exception_handler(DataBackendRequestError)
async def handle_data_backend_error(_: Request, exc: DataBackendRequestError) -> JSONResponse:
    request_id = getattr(_.state, "request_id", "")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "details": exc.details,
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    _logger.exception("Unhandled application error. request_id=%s", request_id)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Erro interno inesperado",
                "request_id": request_id,
            }
        },
    )


@app.get(
    "/health",
    tags=["system"],
    summary="Health check",
    description="Verifica se a API está ativa e retorna metadados básicos de execução.",
)
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "env": settings.app_env,
        "model": settings.openai_model,
    }


app.include_router(profiles_router)
app.include_router(suggestions_router)
