import numpy as np
import pandas as pd
import pytest
from typing import Callable, List

from sklearn.pipeline import Pipeline

from calidhayte.calibrate import Calibrate


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
    coeffs = np.random.randn(2)

    y_df = pd.DataFrame()
    modded = x_df * coeffs

    y_df['x'] = modded.sum(axis=1)

    z_df = pd.DataFrame()
    z_df['x'] = pd.Series(np.random.rand(300))
    z_df['a'] = pd.Series(np.random.rand(300))

    return {
            'x': x_df,
            'y': y_df,
            'z': z_df
            }


@pytest.mark.parametrize("folds", [2, 3, 4, 5])
@pytest.mark.cal
def test_data_split(full_data, folds):
    """
    Tests whether data is split properly
    """
    tests = dict()
    print(full_data['x'])
    print(full_data['y'])
    coeff_inst = Calibrate(
            x_data=full_data['x'],
            y_data=full_data['y'],
            target='x',
            folds=folds
            )
    split_coeffs = coeff_inst.return_measurements()['y']
    num_of_folds = split_coeffs.loc[:, 'Fold'].nunique()
    print(num_of_folds)
    test_prop = split_coeffs.loc[:, 'Fold'].value_counts()[0] /\
        split_coeffs.shape[0]
    print(test_prop)

    tests['Correct Num of Folds'] = num_of_folds == folds
    tests['Correct Prop of Folds'] = test_prop == pytest.approx(1/(folds), 0.1)
    for test, result in tests.items():
        print(f"{test}: {result}")

    assert all(tests.values())


@pytest.mark.cal
@pytest.mark.parametrize("rsearch", [True, False])
def test_skl_cals(full_data, rsearch):
    """
    Combines all possible multivariate key combos with each skl calibration
    method except omp which needs at least 1 mv key
    """
    tests = dict()
    funcs: List[Callable[..., None]] = [
            Calibrate.bayesian_ard,
            Calibrate.bayesian_ridge,
            Calibrate.decision_tree,
            Calibrate.elastic_net,
            Calibrate.elastic_net_cv,
            Calibrate.extra_tree,
            Calibrate.extra_trees_ensemble,
            Calibrate.gaussian_process,
            Calibrate.gradient_boost_regressor,
            Calibrate.hist_gradient_boost_regressor,
            Calibrate.huber,
            Calibrate.isotonic,
            Calibrate.lars,
            Calibrate.lars_lasso,
            Calibrate.lasso,
            Calibrate.lasso_cv,
            Calibrate.linear_svr,
            Calibrate.linreg,
            Calibrate.mlp_regressor,
            Calibrate.nu_svr,
            Calibrate.omp,
            Calibrate.passive_aggressive,
            #  Calibrate.pls,
            Calibrate.random_forest,
            Calibrate.ransac,
            Calibrate.ridge,
            Calibrate.ridge_cv,
            Calibrate.stochastic_gradient_descent,
            Calibrate.svr,
            Calibrate.theil_sen,
            Calibrate.tweedie,
            Calibrate.xgboost,
            Calibrate.xgboost_rf,
            # Calibrate.pymc_bayesian
            ]
    coeff_inst = Calibrate(
            x_data=full_data['x'],
            y_data=full_data['y'],
            target='x'
            )
    for func in funcs:
        print(func)
        func(coeff_inst, random_search=rsearch)
    models = coeff_inst.return_models()
    tests['Correct number of techniques'] = len(
            models.keys()
        ) == len(funcs)
    for technique, scaling_methods in models.items():
        tests[f'Correct number of scalers {technique}'] = len(
                scaling_methods.keys()
                ) == 1
        for _, var_combos in scaling_methods.items():
            if "Orthogonal Matching Pursuit" in technique:
                correct_num_of_vars = 1
            elif "Isotonic Regression" in technique:
                print(var_combos.keys())
                correct_num_of_vars = 1
            else:
                correct_num_of_vars = 2

            tests[f'Correct number of vars {technique}'] = len(
                    var_combos.keys()
                    ) == correct_num_of_vars

            for vars, folds in var_combos.items():
                if (
                    "Cross Validated" not in technique and
                    "(Random Search)" not in technique
                ):
                    tests[
                            f'Correct number of folds {technique} {vars}'
                            ] = len(folds.keys()) == 5
                else:
                    tests[
                            f'Correct number of folds {technique} {vars}'
                            ] = len(folds.keys()) == 1

                for fold, pipe in folds.items():
                    _ = pipe.predict(full_data['z'])
                    tests[
                            f'Pipe for {technique} {vars} {fold} works'
                            ] = isinstance(pipe, Pipeline)

    for test, result in tests.items():
        if not result:
            print(f"{test}: {result}")
    assert all(tests.values())
