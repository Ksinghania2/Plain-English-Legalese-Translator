"""FastAPI application for the Legalese Translator web UI + JSON API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from legalese.translator import translate

app = FastAPI(
    title="Legalese Translator",
    description="Translate dense legal/regulatory text into plain English.",
    version="0.1.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Document or clause to translate.")
    domain: str | None = Field(default=None, description="Optional domain filter.")
    include_entities: bool = Field(default=True)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/translate")
def api_translate(req: TranslateRequest) -> dict:
    try:
        result = translate(req.text, domain=req.domain, include_entities=req.include_entities)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc))
    return result.to_dict()


@app.get("/")
def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not built")
    return FileResponse(index_path)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
