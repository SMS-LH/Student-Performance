from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.feature_engineering import build_preprocessor

def test_linear_regression_performance(sample_raw_data):
    X = sample_raw_data.drop('Exam_Score', axis=1)
    y = sample_raw_data['Exam_Score']
    preprocessor = build_preprocessor()
    X_prep = preprocessor.fit_transform(X)
    model = LinearRegression()
    model.fit(X_prep, y)
    y_pred = model.predict(X_prep)
    r2 = r2_score(y, y_pred)
    assert r2 > 0.0  # doit expliquer quelque chose