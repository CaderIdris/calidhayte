import numpy as np
import pandas as pd
import pytest
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
    for index, var in enumerate(['x', 'a', 'b', 'c']):
        coefficients.loc['x + a + b + c', f'coeff.{var}'] = coeffs[index]
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


@pytest.fixture
def full_data_pymc():
    """
    """
    np.random.seed(8)
    df = pd.DataFrame()
    df['x'] = pd.Series(np.random.rand(300))
    df['a'] = pd.Series(np.random.rand(300))
    df['b'] = pd.Series(np.random.rand(300))
    df['c'] = pd.Series(np.random.rand(300))
    coeffs = np.random.randn(4)

    coefficients = pd.DataFrame()
    for index, var in enumerate(['x', 'a', 'b', 'c']):
        coefficients.loc['x + a + b + c', f'coeff.{var}'] = coeffs[index]
        coefficients.loc['x + a + b + c', f'sd.{var}'] = coeffs[index] / 10
    coefficients.loc['x + a + b + c', 'i.intercept'] = 1
    coefficients.loc['x + a + b + c', 'sd.intercept'] = 0.1

    modded = df * coeffs

    df['y'] = modded.sum(axis=1)
    train, test = ttsplit(
            df,
            test_size=0.5,
            random_state=70,
            shuffle=True
            )
    return train, test, coefficients


@pytest.mark.parametrize("evs", [Results.explained_variance_score, None])
@pytest.mark.parametrize("max", [Results.max, None])
@pytest.mark.parametrize("mabs", [Results.mean_absolute, None])
@pytest.mark.parametrize("rms", [Results.root_mean_squared, None])
@pytest.mark.parametrize("rmsl", [Results.root_mean_squared_log, None])
@pytest.mark.parametrize("medabs", [Results.median_absolute, None])
@pytest.mark.parametrize("map", [Results.mean_absolute_percentage, None])
@pytest.mark.parametrize("r2", [Results.r2, None])
@pytest.mark.parametrize("dset", ["skl", "pymc"])
@pytest.mark.results
def test_result_calcs(
        evs,
        max,
        mabs,
        rms,
        rmsl,
        medabs,
        map,
        r2,
        dset,
        full_data_skl,
        full_data_pymc
        ):
    """
    Tests whether all datasets are selected properly
    """
    tests = dict()
    error_tests = list(
            filter(
                bool,
                [
                    evs,
                    max,
                    mabs,
                    rms,
                    medabs,
                    map,
                    r2,
                    ]
                )
            )
    expected_num_of_cols = len(error_tests)
    if dset == "skl":
        data = full_data_skl
    else:
        data = full_data_pymc
    res = Results(
        train=data[0],
        test=data[1],
        coefficients=data[2],
        datasets_to_use=[
            "Calibrated Train",
            "Calibrated Test",
            "Calibrated Full",
            "Uncalibrated Train",
            "Uncalibrated Test",
            "Uncalibrated Full",
            "Minimum",
            "Maximum"
                ]
        )
    for error_test in error_tests:
        error_test(res)

    # skl data has some negatives so expect rmsl to fail
    if bool(rmsl) and dset == "skl":
        with pytest.raises(
                ValueError,
                match=r"^Mean Squared Logarithmic Error cannot be used when "
                      r"targets contain negative values"
                      ):
            rmsl(res)

    errors = res.return_errors()
    if not error_tests:
        expected_num_of_dfs = 0
    elif dset == "skl":
        expected_num_of_dfs = 6
    else:
        expected_num_of_dfs = 12
    tests['Correct number of dfs'] = len(errors.keys()) == expected_num_of_dfs

    for df_name, df in errors.items():
        print(df_name)
        tests[f'Correct num of cols ({df_name})'] = (
                df.shape[1] == expected_num_of_cols
                )
        tests[f'Correct num of rows ({df_name})'] = df.shape[0] == 1

    assert all(tests.values())
