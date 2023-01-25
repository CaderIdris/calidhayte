import pytest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as ttsplit

from calidhayte import Results


@pytest.fixture
def full_data_skl():
    """
    """
    np.random.seed(4)
    df = pd.DataFrame()
    df['x'] = pd.Series(np.random.rand(300))
    df['a'] = pd.Series(np.random.rand(300))
    df['b'] = pd.Series(np.random.rand(300))
    df['c'] = pd.Series(np.random.rand(300))
    coeffs = np.random.randn(4)

    coefficients = pd.DataFrame()
    for index, var in enumerate(['coeff.x', 'coeff.a', 'coeff.b', 'coeff.c']):
        coefficients.loc['x + a + b + c', var] = coeffs[index]
    coefficients.loc['x + a + b + c', 'i.intercept'] = 1

    modded = df * coeffs

    df['y'] = modded.sum(axis=1)
    train, test = ttsplit(
            df,
            test_size=0.5,
            random_state=68,
            shuffle=True
            )
    return train, test, coefficients

@pytest.mark.parametrize("cal_train", ["", "Calibrated Train"])
@pytest.mark.parametrize("cal_test", ["", "Calibrated Test"])
@pytest.mark.parametrize("cal_full", ["", "Calibrated Full"])
@pytest.mark.parametrize("ucal_train", ["", "Uncalibrated Train"])
@pytest.mark.parametrize("ucal_test", ["", "Uncalibrated Test"])
@pytest.mark.parametrize("ucal_full", ["", "Uncalibrated Full"])
@pytest.mark.parametrize("min", ["", "Minimum"])
@pytest.mark.parametrize("max", ["", "Maximum"])
@pytest.mark.results
def test_prepare_datasets(
        cal_train,
        cal_test,
        cal_full,
        ucal_train,
        ucal_test,
        ucal_full,
        min,
        max,
        full_data_skl
        ):
    tests = [
        cal_train,
        cal_test,
        cal_full,
        ucal_train,
        ucal_test,
        ucal_full,
        min,
        max
        ]
    tests_to_use = filter(bool, tests)

    res = Results(
        train=full_data_skl[0],
        test=full_data_skl[1],
        coefficients=full_data_skl[2],
        test_sets=list(tests_to_use)
        )

    print(res._datasets)

    assert 1 == 1
