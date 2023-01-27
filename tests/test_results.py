import re

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


@pytest.mark.parametrize("cal_train", ["", "Calibrated Train"])
@pytest.mark.parametrize("cal_test", ["", "Calibrated Test"])
@pytest.mark.parametrize("cal_full", ["", "Calibrated Full"])
@pytest.mark.parametrize("ucal_train", ["", "Uncalibrated Train"])
@pytest.mark.parametrize("ucal_test", ["", "Uncalibrated Test"])
@pytest.mark.parametrize("ucal_full", ["", "Uncalibrated Full"])
@pytest.mark.parametrize("min", ["", "Minimum"])
@pytest.mark.parametrize("max", ["", "Maximum"])
@pytest.mark.parametrize("dset", ["skl", "pymc"])
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
        dset,
        full_data_skl,
        full_data_pymc
        ):
    """
    Tests whether all datasets are selected properly
    """
    tests = dict()
    test_sets = [
        cal_train,
        cal_test,
        cal_full,
        ucal_train,
        ucal_test,
        ucal_full,
        min,
        max
        ]
    expected_keys: list[str] = list()
    # Get expected uncalibrated keys
    expected_keys.extend(
        filter(lambda x: bool(re.search(r'^Uncalibrated', x)), test_sets)
            )
    test_sets_to_use = filter(bool, test_sets)
    if dset == "skl":
        data = full_data_skl
        expected_keys.extend(
            filter(lambda x: bool(re.search(r'^Calibrated', x)), test_sets)
                )
    else:
        data = full_data_pymc
        cal_keys = filter(
                lambda x: bool(re.search(r'^Calibrated', x)), test_sets
                )
        min_max_keys = filter(
                lambda x: bool(re.search(r'^Minimum|^Maximum', x)), test_sets
                )
        for cal_key in cal_keys:
            for pymc_subset in ['Mean'] + list(min_max_keys):
                expected_keys.append(f'{cal_key} ({pymc_subset})')

    res = Results(
        train=data[0],
        test=data[1],
        coefficients=data[2],
        datasets_to_use=list(test_sets_to_use)
        )

    returned_dsets = res._datasets

    # Determine whether correct number of keys present
    tests['Correct number of keys'] = (
           len(expected_keys) == len(returned_dsets.keys())
            )

    tests['Correct keys'] = all(
            [key in returned_dsets.keys() for key in expected_keys]
            )

    # Determine dframes are equal sizes and expected size
    for returned_key, dframes in returned_dsets.items():
        tests[f'x and y same shape: {returned_key}'] = (
                dframes['x'].shape[0] == dframes['y'].shape[0]
                )
        tests[f'x and y same index: {returned_key}'] = (
                dframes['x'].index.equals(dframes['y'].index)
                )
        expected_shape = (
                data[0].shape[0] + data[1].shape[0]
                ) if bool(
                        re.search(r'Full', returned_key)
                        ) else data[0].shape[0]
        tests[f'x expected shape: {returned_key}'] = (
                dframes['x'].shape[0] == expected_shape
                )
        tests[f'y expected shape: {returned_key}'] = (
                dframes['y'].shape[0] == expected_shape
                )

    for testkey, result in tests.items():
        print(f"{testkey}: {result}")

    assert all(tests.values())


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
    error_tests = filter(
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

    res.return_errors()
