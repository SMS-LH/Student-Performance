from src.pipeline import run_training_pipeline

def test_pipeline_integration(sample_raw_data):
    results = run_training_pipeline(sample_raw_data)
    assert 'MAE_test' in results
    assert isinstance(results['MAE_test'], float)
    assert isinstance(results['R2_test'], float)
    # Sur des données synthétiques, le R² peut être négatif ; on vérifie seulement qu'il est calculé
    assert -1.0 <= results['R2_test'] <= 1.0