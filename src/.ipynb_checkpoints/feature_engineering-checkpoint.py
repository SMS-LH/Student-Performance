"""Construction du préprocesseur (ColumnTransformer)."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor() -> ColumnTransformer:
    """Retourne le préprocesseur complet (num, ord, nom)."""

    numeric_features = [
        'Hours_Studied', 'Attendance', 'Sleep_Hours',
        'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity'
    ]

    ordinal_features = [
        'Parental_Involvement',
        'Access_to_Resources',
        'Motivation_Level',
        'Family_Income',
        'Teacher_Quality',
        'Peer_Influence',
        'Parental_Education_Level',
        'Distance_from_Home'
    ]

    ordinal_categories = [
        ['Low', 'Medium', 'High'],
        ['Low', 'Medium', 'High'],
        ['Low', 'Medium', 'High'],
        ['Low', 'Medium', 'High'],
        ['Low', 'Medium', 'High'],
        ['Negative', 'Neutral', 'Positive'],
        ['High School', 'College', 'Postgraduate'],
        ['Near', 'Moderate', 'Far']
    ]

    nominal_features = [
        'Extracurricular_Activities', 'Internet_Access',
        'School_Type', 'Learning_Disabilities', 'Gender'
    ]

    numeric_transformer = Pipeline([('scaler', StandardScaler())])

    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories))
    ])

    nominal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('nom', nominal_transformer, nominal_features)
    ], remainder='drop')

    return preprocessor