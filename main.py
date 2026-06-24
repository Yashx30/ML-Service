# app/main.py
from fastapi import FastAPI, Request
import os


# 1. Environment Configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If dotenv not installed or .env missing, continue silently
    pass

DS_MODEL_VERSION = os.getenv("DS_MODEL_VERSION", "ds-v0.1.0")
DS_USE_STUB = os.getenv("DS_USE_STUB", "true").lower() == "true"

# 2. FastAPI Application Factory
def create_app() -> FastAPI:
    app = FastAPI(
        title="ReplyQuick Dental Scan ML Service",
        version=DS_MODEL_VERSION,
        description="FastAPI microservice for dental scan analysis (AI triage + quality scoring)."
    )

   
    # Middleware — pass through request ID (for backend tracing)
    
    @app.middleware("http")
    async def add_request_id_header(request: Request, call_next):
        request_id = request.headers.get("x-request-id")
        response = await call_next(request)
        if request_id:
            response.headers["x-request-id"] = request_id
        return response

    
    # Health Check Endpoint

    @app.get("/healthz")
    async def healthz():
        """
        Simple health check endpoint for uptime and config validation.
        """
        return {
            "ok": True,
            "version": DS_MODEL_VERSION,
            "stub_mode": DS_USE_STUB
        }

    # Mount the Analyze Router (POST + GET)

    from app.routers.analyze import router as analyze_router
    app.include_router(analyze_router)

    return app


# 3. Application Instance

app = create_app()


# 4. Local Development Entry Point

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
  