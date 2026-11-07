from ai4cyber.api import app


def test_api_surface_is_ml_only():
    paths = set(app.openapi()["paths"])
    assert {"/", "/health", "/model", "/predict"}.issubset(paths)
    assert "/chat" not in paths
