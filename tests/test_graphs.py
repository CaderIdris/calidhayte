import re

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split as ttsplit

from calidhayte import Graphs


@pytest.fixture
def full_data_skl():
    """
    """
    np.random.seed(4)
    df = pd.DataFrame()
    df['x'] = pd.Series(np.random.rand(300))
    df['a'] = pd.Series(np.random.rand(300))
    df['b'] = pd.Series(np.random.rand(300))
    coeff_nums = np.random.randn(3)
    coeffs = {
            'x': coeff_nums[0],
            'a': coeff_nums[1],
            'b': coeff_nums[2]
            }

    coefficients = pd.DataFrame()
    combos = [
            ['x'], ['a'], ['b'],
            ['x', 'a'], ['x', 'b'], ['a', 'b'],
            ['x', 'a', 'b']
            ]
    for combo in combos:
        for var in combo:
            coefficients.loc[' + '.join(combo), f'coeff.{var}'] = coeffs[var]
        coefficients.loc[' + '.join(combo), 'i.intercept'] = 1

    modded = df * coeff_nums

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
    coeff_nums = np.random.randn(3)
    coeffs = {
            'x': coeff_nums[0],
            'a': coeff_nums[1],
            'b': coeff_nums[2]
            }

    coefficients = pd.DataFrame()
    combos = [
            ['x'], ['a'], ['b'],
            ['x', 'a'], ['x', 'b'], ['a', 'b'],
            ['x', 'a', 'b']
            ]
    for combo in combos:
        for var in combo:
            coefficients.loc[' + '.join(combo), f'coeff.{var}'] = coeffs[var]
            coefficients.loc[' + '.join(combo), f'sd.{var}'] = coeffs[var] / 10
        coefficients.loc[' + '.join(combo), 'i.intercept'] = 1
        coefficients.loc[' + '.join(combo), 'sd.intercept'] = 0.1

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
@pytest.mark.graphs
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
        min_max_keys = list(
            filter(
                lambda x: bool(re.search(r'^Minimum|^Maximum', x)), test_sets
                )
            )
        for cal_key in cal_keys:
            for pymc_subset in ['Mean'] + min_max_keys:
                expected_keys.append(f'{cal_key} ({pymc_subset})')

    graphs = Graphs(
        train=data[0],
        test=data[1],
        coefficients=data[2],
        datasets_to_use=list(test_sets_to_use)
        )

    returned_dsets = graphs._datasets

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


@pytest.mark.parametrize("lreg", [Graphs.linear_reg_plot, None])
@pytest.mark.parametrize("ba", [Graphs.bland_altman_plot, None])
@pytest.mark.parametrize("ecdf", [Graphs.ecdf_plot, None])
@pytest.mark.parametrize("ts", [Graphs.time_series_plot, None])
@pytest.mark.parametrize("dset", ["skl", "pymc"])
@pytest.mark.graphplots
def test_plot(
        lreg,
        ba,
        ecdf,
        ts,
        dset,
        full_data_skl,
        full_data_pymc
        ):
    tests = dict()
    if dset == "skl":
        data = full_data_skl
        expected_keys = 6
    else:
        data = full_data_pymc
        expected_keys = 12
    plot_tests = list(
            filter(
                bool,
                [
                    lreg,
                    ba,
                    ecdf,
                    ts
                    ]
                )
            )
    graphs = Graphs(
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
    for plot_test in plot_tests:
        plot_test(graphs)
    plot_dict = graphs._plots

    if len(plot_tests) == 1 and plot_tests[0] is ts:
        expected_keys = expected_keys // 3

    if len(plot_tests):
        tests['Correct number of keys'] = len(plot_dict.keys()) == expected_keys
    else:
        tests['Nothing happens'] = True
    for subset, vars in plot_dict.items():
        if bool(re.search(r'^Uncalibrated ', subset)):
            expected_vars = 1
        elif len(plot_tests) == 1 and plot_tests[0] is lreg:
            expected_vars = 3
        elif len(plot_tests) == 2 and all([
                x in [lreg, ts] for x in plot_tests
        ]) and not bool(re.search(r' Full', subset)):
            expected_vars = 3
        else:
            expected_vars = 7
        tests[f'Correct number of vars ({subset})'] = (
                len(vars.keys()) == expected_vars
                )

    for test, result in tests.items():
        print(f'{test}: {result}')
    assert all(tests.values())
