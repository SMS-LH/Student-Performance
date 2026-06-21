import pandas as pd
from src.data_preprocessing import cap_exam_score

def test_cap_exam_score(sample_raw_data):
    df = sample_raw_data.copy()
    df_capped = cap_exam_score(df, max_score=100)
    assert (df_capped['Exam_Score'] <= 100).all()
    assert df_capped['Exam_Score'].iloc[0] == 100  # 105 -> 100