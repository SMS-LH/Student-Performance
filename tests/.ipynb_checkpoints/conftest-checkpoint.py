import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def sample_raw_data():
    """Crée un petit DataFrame réaliste pour tester le preprocessing."""
    np.random.seed(42)
    n = 100
    data = {
        'Hours_Studied': np.random.randint(1, 44, n),
        'Attendance': np.random.randint(60, 101, n),
        'Parental_Involvement': np.random.choice(['Low', 'Medium', 'High'], n),
        'Access_to_Resources': np.random.choice(['Low', 'Medium', 'High'], n),
        'Extracurricular_Activities': np.random.choice(['No', 'Yes'], n),
        'Sleep_Hours': np.random.randint(4, 11, n),
        'Previous_Scores': np.random.randint(50, 101, n),
        'Motivation_Level': np.random.choice(['Low', 'Medium', 'High'], n),
        'Internet_Access': np.random.choice(['No', 'Yes'], n),
        'Tutoring_Sessions': np.random.randint(0, 9, n),
        'Family_Income': np.random.choice(['Low', 'Medium', 'High'], n),
        'Teacher_Quality': np.random.choice(['Low', 'Medium', 'High'], n),
        'School_Type': np.random.choice(['Public', 'Private'], n),
        'Peer_Influence': np.random.choice(['Negative', 'Neutral', 'Positive'], n),
        'Physical_Activity': np.random.randint(0, 7, n),
        'Learning_Disabilities': np.random.choice(['No', 'Yes'], n),
        'Parental_Education_Level': np.random.choice(
            ['High School', 'College', 'Postgraduate'], n),
        'Distance_from_Home': np.random.choice(['Near', 'Moderate', 'Far'], n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Exam_Score': np.random.normal(67, 4, n).round(0)
    }
    df = pd.DataFrame(data)
    df.loc[0, 'Exam_Score'] = 105  # pour tester le plafonnement
    return df