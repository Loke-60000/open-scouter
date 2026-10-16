"""
Pydantic request/response models for the HTTP API.
Chat endpoints follow the OpenAI schema so existing clients work unchanged; the
assistant-axis monitor is attached as an extra ``axis_monitor`` field on
responses (OpenAI clients ignore unknown fields).
"""

from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessageModel(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessageModel]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    stop: list[str] = Field(default_factory=list)
    seed: Optional[int] = None
    stream: bool = False
    # Per-request overrides (omit to use the server's live settings).
    monitor: Optional[bool] = None


class AxisMonitorModel(BaseModel):
    layer: int
    count: int
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    last: Optional[float] = None
    drift: Optional[float] = None
    scores: list[float] = Field(default_factory=list)


class FeatureHitModel(BaseModel):
    index: int
    label: str
    activation: float
    max_activation: float


class ChatChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: str = "stop"


class UsageModel(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: UsageModel
    # open-scouter extensions:
    axis_monitor: Optional[AxisMonitorModel] = None
    emotion_monitor: Optional[dict[str, float]] = None
    feature_monitor: Optional[list[FeatureHitModel]] = None


class SteeringModel(BaseModel):
    mode: Literal["off", "addition", "capping"]
    coefficient: float = 0.0
    cap_threshold: float = 0.0
    layers: list[int] = Field(default_factory=list)
    active: bool = False


class SteeringUpdate(BaseModel):
    mode: Optional[Literal["off", "addition", "capping"]] = None
    coefficient: Optional[float] = None
    cap_threshold: Optional[float] = None
    layers: Optional[list[int]] = None


class AxisInfo(BaseModel):
    loaded: bool
    source: Optional[str] = None
    n_layers: Optional[int] = None
    hidden_dim: Optional[int] = None
    target_layer: Optional[int] = None


class AxisLoadRequest(BaseModel):
    # Local .pt/.npz path, "hf:<model-key>", "feature:<index>" or "emotion:<name>".
    spec: str
    target_layer: Optional[int] = None


class EmotionsInfo(BaseModel):
    loaded: bool
    count: int = 0
    names: list[str] = Field(default_factory=list)


class EmotionsLoadRequest(BaseModel):
    # Local .npz path produced by `scouter-emotions`.
    spec: str


class FeaturesInfo(BaseModel):
    loaded: bool
    count: int = 0
    layer: Optional[int] = None
    labeled: int = 0


class FeaturesLoadRequest(BaseModel):
    # Local .npz path produced by `scouter-features`.
    spec: str


class FeatureInfo(BaseModel):
    index: int
    label: str
    max_activation: float
    # `addition` coefficient that clamps the feature to 10x its max (Golden Gate recipe).
    clamp_coefficient: float


class MonitorRequest(BaseModel):
    """Run a (non-streaming) generation and return only the monitor trace."""

    messages: list[ChatMessageModel]
    max_tokens: int = 128
    temperature: float = 0.7


class HealthResponse(BaseModel):
    status: str
    backend: str
    model: str
    model_loaded: bool
    axis_loaded: bool
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    steering: SteeringModel
