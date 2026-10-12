"""FastAPI inference service.

Exposes the full pipeline as a single endpoint. The service takes a
preprocessed volume, lesion candidates, and optional prior and
radiologist inputs, and returns a structured response: priority tier,
temporal changes, discrepancy flag, and safety recommendation.

The service is deliberately thin. It does not load models or run
inference itself. It receives a precomputed `PipelineResult` from a
caller-injected pipeline runner and serializes it. This keeps the
service testable without a trained model and keeps the serving layer
separate from the inference layer.

This is a research prototype. Every response carries an explicit
non-clinical framing field. That field is not optional and is not
removable by configuration.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from imagex.pipeline import (
    PerceptionOutput,
    PipelineResult,
    pipeline_result_to_dict,
    run_pipeline,
)
from imagex.temporal.matching import Lesion


RESEARCH_PROTOTYPE_FRAMING = (
    "Research prototype. Not for clinical decision-making. "
    "Every output must be reviewed by a qualified radiologist."
)


class LesionIn(BaseModel):
    lesion_id: str
    centroid_mm: tuple[float, float, float]
    volume_mm3: float = Field(ge=0.0)
    diameter_mm: float = Field(gt=0.0)


class PipelineRequest(BaseModel):
    abnormality_score: float = Field(ge=0.0, le=1.0)
    lesion_candidates: list[LesionIn]
    safety_recommendation: str
    uncertainty: float = Field(ge=0.0)
    is_ood: bool
    radiologist_lesions: list[LesionIn] | None = None


class PipelineResponse(BaseModel):
    framing: str
    result: dict[str, Any]


def _to_lesions(items: list[LesionIn]) -> list[Lesion]:
    return [
        Lesion(
            lesion_id=it.lesion_id,
            centroid_mm=tuple(it.centroid_mm),
            volume_mm3=it.volume_mm3,
            diameter_mm=it.diameter_mm,
        )
        for it in items
    ]


def create_app(
    pipeline_runner: Callable[..., PipelineResult] = run_pipeline,
) -> FastAPI:
    """Build the FastAPI app with an injected pipeline runner.

    Injection lets tests substitute a deterministic runner without
    requiring a trained model or real inference.
    """
    app = FastAPI(
        title="IMAGEX inference service",
        version="0.0.1-research-prototype",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "framing": RESEARCH_PROTOTYPE_FRAMING}

    @app.post("/predict", response_model=PipelineResponse)
    def predict(req: PipelineRequest) -> PipelineResponse:
        try:
            perception = PerceptionOutput(
                abnormality_score=req.abnormality_score,
                lesions=_to_lesions(req.lesion_candidates),
            )
            rad = (
                None
                if req.radiologist_lesions is None
                else _to_lesions(req.radiologist_lesions)
            )
            result = pipeline_runner(
                perception=perception,
                safety_recommendation=req.safety_recommendation,
                uncertainty=req.uncertainty,
                is_ood=req.is_ood,
                radiologist_lesions=rad,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return PipelineResponse(
            framing=RESEARCH_PROTOTYPE_FRAMING,
            result=pipeline_result_to_dict(result),
        )

    return app


app = create_app()
