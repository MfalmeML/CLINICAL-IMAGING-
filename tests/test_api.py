"""Tests for the FastAPI serving layer."""

from fastapi.testclient import TestClient

from imagex.serving.api import RESEARCH_PROTOTYPE_FRAMING, create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_health_returns_framing() -> None:
    r = _client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "Research prototype" in body["framing"]


def test_predict_minimal_request() -> None:
    r = _client().post(
        "/predict",
        json={
            "abnormality_score": 0.2,
            "lesion_candidates": [],
            "safety_recommendation": "safe-to-predict",
            "uncertainty": 0.0,
            "is_ood": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "Research prototype" in body["framing"]
    assert body["result"]["priority"] == "LOW"
    assert body["result"]["temporal_changes"] is None
    assert body["result"]["discrepancy"] is None


def test_predict_with_lesions_and_radiologist() -> None:
    r = _client().post(
        "/predict",
        json={
            "abnormality_score": 0.85,
            "lesion_candidates": [
                {
                    "lesion_id": "a1",
                    "centroid_mm": [0.0, 0.0, 0.0],
                    "volume_mm3": 65.0,
                    "diameter_mm": 12.0,
                }
            ],
            "safety_recommendation": "safe-to-predict",
            "uncertainty": 0.0,
            "is_ood": False,
            "radiologist_lesions": [
                {
                    "lesion_id": "r1",
                    "centroid_mm": [1.0, 0.0, 0.0],
                    "volume_mm3": 65.0,
                    "diameter_mm": 12.0,
                }
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["priority"] in ("HIGH", "URGENT")
    assert body["result"]["discrepancy"] is not None


def test_predict_rejects_bad_safety_recommendation() -> None:
    r = _client().post(
        "/predict",
        json={
            "abnormality_score": 0.2,
            "lesion_candidates": [],
            "safety_recommendation": "maybe",
            "uncertainty": 0.0,
            "is_ood": False,
        },
    )
    assert r.status_code == 400


def test_predict_rejects_out_of_range_abnormality() -> None:
    r = _client().post(
        "/predict",
        json={
            "abnormality_score": 1.5,
            "lesion_candidates": [],
            "safety_recommendation": "safe-to-predict",
            "uncertainty": 0.0,
            "is_ood": False,
        },
    )
    assert r.status_code == 422


def test_framing_is_present_on_every_response() -> None:
    client = _client()
    for endpoint in ("/health",):
        body = client.get(endpoint).json()
        assert "Research prototype" in body["framing"]
    body = client.post(
        "/predict",
        json={
            "abnormality_score": 0.5,
            "lesion_candidates": [],
            "safety_recommendation": "safe-to-predict",
            "uncertainty": 0.0,
            "is_ood": False,
        },
    ).json()
    assert body["framing"] == RESEARCH_PROTOTYPE_FRAMING
