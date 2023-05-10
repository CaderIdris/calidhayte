import numpy as np
import pandas as pd
import pytest

from calidhayte import Calibrate


@pytest.fixture
def full_data():
    """
    Dataset composed of random values. y df constructed from x_df scaled by
    4 random coeffs
    """
    np.random.seed(72)
    x_df = pd.DataFrame()
    x_df['x'] = pd.Series(np.random.rand(300))
    x_df['a'] = pd.Series(np.random.rand(300))
    x_df['b'] = pd.Series(np.random.rand(300))
    x_df['c'] = pd.Series(np.random.rand(300))
    coeffs = np.random.randn(4)

    y_df = pd.DataFrame()
    modded = x_df * coeffs

    y_df['x'] = modded.sum(axis=1)

    return {
            'x': x_df,
            'y': y_df
            }


@pytest.mark.parametrize("folds", [2, 3, 4, 5])
def test_data_split(full_data, folds):
    """
    Tests whether data is split properly
    """
    tests = dict()
    coeff_inst = Calibrate(
            x_data=full_data['x'],
            y_data=full_data['y'],
            target='x'
            )
    split_coeffs = coeff_inst.return_measurements()['y']
    num_of_folds = split_coeffs.loc[:, 'Fold'].nunique().size
    test_prop = split_coeffs.loc[:, 'Fold'].nunique()[0] /\
        split_coeffs.shape[0]

    tests['Correct Num of Folds'] = num_of_folds == folds
    tests['Correct Prop of Folds'] = test_prop == pytest.approx(1/(folds), 0.1)
    for test, result in tests.items():
        print(f"{test}: {result}")

    assert all(tests.values())


@pytest.mark.parametrize("reg_func",
                         [
                             Coefficients.ols,
                             Coefficients.ridge,
                             Coefficients.lasso,
                             Coefficients.elastic_net,
                             Coefficients.lars,
                             Coefficients.lasso_lars,
                             Coefficients.ransac,
                             Coefficients.theil_sen
                             ]
                         )
@pytest.mark.parametrize("mv_keys",
                         [
                             ([[]]),
                             ([["a"]]),
                             ([["b"]]),
                             ([["c"]]),
                             ([["a", "b"]]),
                             ([["a", "c"]]),
                             ([["b", "c"]]),
                             ([["a", "b", "c"]]),
                             ([
                                 [],
                                 ["a"],
                                 ["b"],
                                 ["c"],
                                 ["a", "b"],
                                 ["a", "c"],
                                 ["b", "c"],
                                 ["a", "b", "c"]
                                 ])
                             ]
                         )
def test_skl_single_cals_ex_omp(full_data, reg_func, mv_keys):
    """
    Combines all possible multivariate key combos with each skl calibration
    method except omp which needs at least 1 mv key
    """
    tests = dict()

    coeff_inst = Coefficients(
            x_data=full_data['x'],
            y_data=full_data['y']
            )
    for keys in mv_keys:
        reg_func(coeff_inst, keys)
    coeffs = coeff_inst.return_coefficients()
    for technique, df in coeffs.items():
        tests[f'Correct number of rows ({technique})'] = (
                df.shape[0] == len(mv_keys)
                )
        tests[f'Correct number of columns ({technique})'] = (
                df.shape[1] == len(mv_keys[-1]) + 2
                )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.parametrize("mv_keys",
                         [
                             ([["a"]]),
                             ([["b"]]),
                             ([["c"]]),
                             ([["a", "b"]]),
                             ([["a", "c"]]),
                             ([["b", "c"]]),
                             ([["a", "b", "c"]]),
                             ([
                                 ["a"],
                                 ["b"],
                                 ["c"],
                                 ["a", "b"],
                                 ["a", "c"],
                                 ["b", "c"],
                                 ["a", "b", "c"]
                                 ])
                             ]
                         )
def test_skl_omp(full_data, mv_keys):
    """
    Combines all possible multivariate key combos with omp calibration
    method
    """
    tests = dict()

    coeff_inst = Coefficients(
            x_data=full_data['x'],
            y_data=full_data['y']
            )
    for keys in mv_keys:
        coeff_inst.orthogonal_matching_pursuit(keys)
    coeffs = coeff_inst.return_coefficients()
    for technique, df in coeffs.items():
        tests[f'Correct number of rows ({technique})'] = (
                df.shape[0] == len(mv_keys)
                )
        tests[f'Correct number of columns ({technique})'] = (
                df.shape[1] == len(mv_keys[-1]) + 2
                )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.parametrize("mv_keys",
                         [
                             ([
                                 ["a"],
                                 ["a", "b"],
                                 ["a", "b", "c"]
                                 ])
                             ]
                         )
@pytest.mark.parametrize("ols", [None, Coefficients.ols])
@pytest.mark.parametrize("ridge", [None, Coefficients.ridge])
@pytest.mark.parametrize("lasso", [None, Coefficients.lasso])
@pytest.mark.parametrize("enet", [None, Coefficients.elastic_net])
@pytest.mark.parametrize("lars", [None, Coefficients.lars])
@pytest.mark.parametrize("laslars", [None, Coefficients.lasso_lars])
@pytest.mark.parametrize("ransac", [None, Coefficients.ransac])
@pytest.mark.parametrize("theilsen", [None, Coefficients.theil_sen])
def test_combo_cal_skl(
        full_data,
        mv_keys,
        ols,
        ridge,
        lasso,
        enet,
        lars,
        laslars,
        ransac,
        theilsen,
        ):
    """
    Tests all combos of skl calibration techniques against several combos of
    multivariate keys
    """
    cals_ex_student = [
            ols,
            ridge,
            lasso,
            enet,
            lars,
            laslars,
            ransac,
            theilsen
            ]
    tests = dict()
    coeff_inst = Coefficients(
            x_data=full_data['x'],
            y_data=full_data['y']
            )

    for cal in cals_ex_student:
        if cal is not None:
            for key in mv_keys:
                cal(coeff_inst, key)

    coeffs = coeff_inst.return_coefficients()

    expected_keys = np.array(cals_ex_student, dtype=bool).sum()
    tests['Correct number of keys in coeffs dict'] = (
            len(coeffs.keys()) == expected_keys
            )

    for technique, df in coeffs.items():
        tests[f'Correct number of rows ({technique})'] = (
                df.shape[0] == len(mv_keys)
                )
        tests[f'Correct number of columns ({technique})'] = (
                df.shape[1] == len(mv_keys[-1]) + 2
                )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.parametrize("mv_keys",
                         [
                             ([
                                 ["a"],
                                 ["a", "b"],
                                 ["a", "b", "c"]
                                 ])
                             ]
                         )
@pytest.mark.parametrize("family", ["Gaussian", "Student T"])
def test_bayesian(full_data, mv_keys, family):
    tests = dict()
    coeff_inst = Coefficients(
            x_data=full_data['x'],
            y_data=full_data['y']
            )
    for key in mv_keys:
        coeff_inst.bayesian(key, family)

    coeffs = coeff_inst.return_coefficients()
    tests['Correct family in key'] = (
            list(coeffs.keys())[0] == f"Bayesian ({family})"
            )

    for technique, df in coeffs.items():
        tests[f'Correct number of rows ({technique})'] = (
                df.shape[0] == len(mv_keys)
                )
        tests[f'Correct number of columns ({technique})'] = (
                df.shape[1] == (len(mv_keys[-1]) + 2) * 2
                )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())
