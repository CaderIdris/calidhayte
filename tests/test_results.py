import numpy as np
import pandas as pd
import pytest

from calidhayte.calibrate import Calibrate
from calidhayte.results import Results


@pytest.fixture
def trained_models():
    """
    """
    np.random.seed(4)
    x_df = pd.DataFrame()
    x_df['x'] = pd.Series(np.random.rand(300))
    x_df['a'] = pd.Series(np.random.rand(300))
    x_df['b'] = pd.Series(np.random.rand(300))
    x_df['c'] = pd.Series(np.random.rand(300))
    coeffs = np.random.randn(4)

    y_df = pd.DataFrame()
    modded = x_df * coeffs

    y_df['x'] = modded.sum(axis=1)
    y_df['Fold'] = ([0]*60) + ([1]*60) + ([2]*60) + ([3]*60) + ([4]*60)

    cal = Calibrate(
            x_df,
            y_df,
            target='x'
            )
    cal.linreg()
    cal.theil_sen()
    cal.random_forest()

    return {
            'x_data': x_df,
            'y_data': y_df,
            'target': 'x',
            'models': cal.return_models()
            }


@pytest.mark.results
def test_prepare_datasets(
        trained_models
        ):
    """
    Tests whether all datasets are selected properly
    """
    tests = dict()

    results = Results(**trained_models)

    errs = [
            Results.explained_variance_score,
            Results.max,
            Results.mean_absolute,
            Results.root_mean_squared,
            Results.median_absolute,
            Results.mean_absolute_percentage,
            Results.r2,
            Results.mean_pinball_loss
            ]
    for err in errs:
        err(results)

    res = results.return_errors()
    tests['Correct num of techniques'] = res.index.to_frame().nunique()[
            'Technique'
            ] == 3
    tests['Correct num of vars'] = res.index.to_frame().nunique()[
            'Variables'
            ] == 8
    tests['Correct num of folds'] = res.index.to_frame().nunique()[
            'Fold'
            ] == 5
    print(res.index.to_frame().nunique())

    tests['Correct num of rows'] = res.shape[0] == 120

    tests['Correct num of cols'] = res.shape[1] == 8

    assert all(tests.values())
