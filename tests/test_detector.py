import numpy as np

from src.safety_glasses_detector import extract_features, predict_safety_glasses


def test_extract_features_returns_expected_shape():
    patch = np.zeros((80, 80), dtype=np.uint8)
    features = extract_features(patch)
    assert len(features) == 5
    assert np.isfinite(features).all()


def test_predict_safety_glasses_distinguishes_plain_and_glass_like_regions():
    plain = np.zeros((80, 80), dtype=np.uint8)
    glass_like = np.full((80, 80), 180, dtype=np.uint8)
    glass_like[:, 20:60] = 80

    plain_score = predict_safety_glasses(plain)
    glass_score = predict_safety_glasses(glass_like)

    assert plain_score < glass_score
