import numpy as np
from src.feature_engineering import build_preprocessor

def test_preprocessor_shape(sample_raw_data):
    X = sample_raw_data.drop('Exam_Score', axis=1)
    preprocessor = build_preprocessor()
    X_prep = preprocessor.fit_transform(X)
    assert X_prep.shape[1] == 19

def test_preprocessor_no_missing(sample_raw_data):
    X = sample_raw_data.drop('Exam_Score', axis=1)
    preprocessor = build_preprocessor()
    X_prep = preprocessor.fit_transform(X)
    assert not np.isnan(X_prep).any()