from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from sse_starlette.sse import EventSourceResponse

from ..backends import ChatMessage, SamplingParams
from ..config import Settings
from ..steering import SteeringConfig
from .schemas import (
    AxisInfo,
    AxisLoadRequest,
    AxisMonitorModel,
    ChatChoice,
    ChatChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageModel,
    EmotionsInfo,
    EmotionsLoadRequest,
    HealthResponse,
    MonitorRequest,
    SteeringModel,
    SteeringUpdate,
    UsageModel,
)
from .state import AppState

logger = logging.getLogger("scouter")


def _to_messages(items: list[ChatMessageModel]) -> list[ChatMessage]:
    return [ChatMessage(role=m.role, content=m.content) for m in items]


def _params_from_request(req: ChatCompletionRequest) -> SamplingParams:
    return SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=list(req.stop or []),
        seed=req.seed,
    )


def _steering_model(cfg: SteeringConfig) -> SteeringModel:
    return SteeringModel(
        mode=cfg.mode,
        coefficient=cfg.coefficient,
        cap_threshold=cfg.cap_threshold,
        layers=list(cfg.layers),
        active=cfg.active,
    )


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state: AppState = app.state.scouter
        # Eager model load can be slow; only do it if explicitly requested via
        # env (SCOUTER_EAGER_LOAD). Otherwise the model loads on first request.
        import os

        if os.getenv("SCOUTER_EAGER_LOAD", "0") == "1":
            await run_in_threadpool(state.ensure_model)
        yield

    app = FastAPI(title="open-scouter", version="0.1.0", lifespan=lifespan)
    app.state.scouter = AppState(settings)

    def state() -> AppState:
        return app.state.scouter

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        st = state()
        return HealthResponse(
            status="ok",
            backend=st.settings.backend,
            model=st.settings.model,
            model_loaded=st.model_loaded,
            axis_loaded=st.axis is not None,
            num_layers=st.backend.num_layers if st.model_loaded else None,
            hidden_size=st.backend.hidden_size if st.model_loaded else None,
            steering=_steering_model(st.steering.get()),
        )

    @app.get("/v1/models")
    async def models():
        st = state()
        return {
            "object": "list",
            "data": [
                {
                    "id": st.settings.served_model_name,
                    "object": "model",
                    "owned_by": "scouter",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        req: ChatCompletionRequest,
    ) -> "ChatCompletionResponse | EventSourceResponse":
        st = state()
        messages = _to_messages(req.messages)
        params = _params_from_request(req)
        cid = f"chatcmpl-{uuid.uuid4().hex}"
        model_name = st.settings.served_model_name

        if req.stream:
            return _stream_response(st, messages, params, req.monitor, cid, model_name)

        result = await run_in_threadpool(st.generate, messages, params, monitor=req.monitor)
        monitor_model = AxisMonitorModel(**result.monitor.to_dict()) if result.monitor else None
        return ChatCompletionResponse(
            id=cid,
            model=model_name,
            choices=[
                ChatChoice(
                    message=ChatChoiceMessage(content=result.text),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=UsageModel(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
            axis_monitor=monitor_model,
            emotion_monitor=result.emotion_profile,
        )

    def _stream_response(
        st: AppState,
        messages: list[ChatMessage],
        params: SamplingParams,
        monitor: Optional[bool],
        cid: str,
        model_name: str,
    ) -> EventSourceResponse:
        async def event_gen() -> AsyncIterator[dict[str, str]]:
            # The backend's stream is a blocking generator; iterate_in_threadpool
            # drives it in a worker thread and yields chunks back on the loop.
            async for chunk in iterate_in_threadpool(st.stream(messages, params, monitor=monitor)):
                delta = {"role": "assistant", "content": chunk.text}
                payload: dict[str, object] = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": chunk.finish_reason}],
                }
                if chunk.score is not None:
                    payload["axis_score"] = chunk.score
                yield {"data": json.dumps(payload)}
            yield {"data": "[DONE]"}

        return EventSourceResponse(event_gen())

    @app.get("/v1/steering", response_model=SteeringModel)
    async def get_steering() -> SteeringModel:
        return _steering_model(state().steering.get())

    @app.post("/v1/steering", response_model=SteeringModel)
    async def set_steering(update: SteeringUpdate) -> SteeringModel:
        st = state()
        # Validate mode against the backend's capabilities.
        if update.mode and update.mode != "off":
            supported = getattr(st.backend, "supported_steer_modes", ("off",))
            if update.mode not in supported:
                raise HTTPException(
                    status_code=400,
                    detail=f"backend {st.backend.name!r} does not support steering "
                    f"mode {update.mode!r}; supported: {list(supported)}",
                )
        if (update.mode and update.mode != "off") and st.axis is None:
            raise HTTPException(
                status_code=400, detail="no axis loaded; load one via POST /v1/axis"
            )
        cfg = st.steering.update(
            mode=update.mode,
            coefficient=update.coefficient,
            cap_threshold=update.cap_threshold,
            layers=update.layers,
        )
        return _steering_model(cfg)

    @app.get("/v1/axis", response_model=AxisInfo)
    async def get_axis() -> AxisInfo:
        st = state()
        ax = st.axis
        if ax is None:
            return AxisInfo(loaded=False)
        return AxisInfo(
            loaded=True,
            source=ax.source,
            n_layers=ax.n_layers,
            hidden_dim=ax.hidden_dim,
            target_layer=st.settings.target_layer if st.settings.target_layer >= 0 else None,
        )

    @app.post("/v1/axis", response_model=AxisInfo)
    async def post_axis(req: AxisLoadRequest) -> AxisInfo:
        st = state()
        await run_in_threadpool(st.ensure_model)
        try:
            ax = await run_in_threadpool(st.load_axis, req.spec, req.target_layer)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"failed to load axis: {exc}") from exc
        return AxisInfo(
            loaded=True,
            source=ax.source,
            n_layers=ax.n_layers,
            hidden_dim=ax.hidden_dim,
            target_layer=st.settings.target_layer,
        )

    @app.post("/v1/monitor")
    async def monitor(req: MonitorRequest) -> dict:
        st = state()
        if st.axis is None and st.emotions is None:
            raise HTTPException(
                status_code=400, detail="nothing to monitor; load an axis or emotions first"
            )
        messages = _to_messages(req.messages)
        params = SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature)
        result = await run_in_threadpool(st.generate, messages, params, monitor=True)
        return {
            "axis_monitor": result.monitor.to_dict() if result.monitor else None,
            "emotion_monitor": result.emotion_profile,
        }

    def _emotions_info() -> EmotionsInfo:
        ev = state().emotions
        if ev is None:
            return EmotionsInfo(loaded=False)
        return EmotionsInfo(loaded=True, count=len(ev), names=ev.names)

    @app.get("/v1/emotions", response_model=EmotionsInfo)
    async def get_emotions() -> EmotionsInfo:
        return _emotions_info()

    @app.post("/v1/emotions", response_model=EmotionsInfo)
    async def post_emotions(req: EmotionsLoadRequest) -> EmotionsInfo:
        st = state()
        await run_in_threadpool(st.ensure_model)
        try:
            await run_in_threadpool(st.load_emotions, req.spec)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"failed to load emotions: {exc}") from exc
        return _emotions_info()

    return app
