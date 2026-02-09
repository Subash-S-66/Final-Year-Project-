from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FrameMeta(BaseModel):
    type: Literal["frame_meta"]
    frame_id: str
    ts_ms: int = Field(..., description="Client-side timestamp in milliseconds")
    encoding: Literal["jpeg", "webp"]
    width: int
    height: int


class PredictionMessage(BaseModel):
    type: Literal["prediction"] = "prediction"
    frame_id: str
    token: str
    confidence: float
    is_committed: bool = False  # True if smoothed/stable, False if live/unstable
    latency_ms: int
    debug: Optional[dict] = None
