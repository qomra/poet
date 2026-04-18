"""
Fine-Tashkeel diacritization service.

Loads basharalrfooh/Fine-Tashkeel (ByT5, classical-Arabic-tuned) onto the
configured CUDA device and exposes a batched FastAPI endpoint.

Endpoints:
    GET  /health          → { status, device, model }
    POST /diacritize      → body: { texts: list[str], max_new_tokens?: int }
                            response: { diacritized: list[str] }
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "basharalrfooh/Fine-Tashkeel")
DEVICE = os.getenv("DEVICE", "cuda:0")
DTYPE = os.getenv("DTYPE", "float16")  # float16 / bfloat16 / float32
DEFAULT_MAX_NEW = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "256"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))

app = FastAPI(title="Fine-Tashkeel service")

_tokenizer = None
_model = None


def _dtype() -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(DTYPE, torch.float32)


@app.on_event("startup")
def _load() -> None:
    global _tokenizer, _model
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=_dtype())
    _model.to(DEVICE)
    _model.eval()


class DiacritizeIn(BaseModel):
    texts: list[str] = Field(..., description="Raw Arabic strings to diacritize")
    max_new_tokens: Optional[int] = Field(None, description="Override max_new_tokens")


class DiacritizeOut(BaseModel):
    diacritized: list[str]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _model is not None else "loading",
        "device": DEVICE,
        "model": MODEL_ID,
        "dtype": DTYPE,
    }


@app.post("/diacritize", response_model=DiacritizeOut)
def diacritize(body: DiacritizeIn) -> DiacritizeOut:
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="model still loading")
    if not body.texts:
        return DiacritizeOut(diacritized=[])
    if len(body.texts) > MAX_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"batch too large; max is {MAX_BATCH}",
        )

    max_new = body.max_new_tokens or DEFAULT_MAX_NEW

    out: list[str] = []
    # ByT5 is byte-level; pad to the longest item in the batch for efficiency
    enc = _tokenizer(body.texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        gen = _model.generate(
            **enc,
            max_new_tokens=max_new,
            num_beams=1,
            do_sample=False,
        )
    for seq in gen:
        out.append(_tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return DiacritizeOut(diacritized=out)
