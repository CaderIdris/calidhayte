import datetime as dt
from pathlib import Path
import pickle
from typing import Callable, List

import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from calidhayte.calibrate import (
    Calibrate,
    TimeColumnTransformer
)


@pytest.fixture()
def full_data():
    """
    Dataset composed of random values. y df constructed from x_df scaled by
    4 random coeffs
    """
    np.random.seed(72)
    x_df = pd.DataFrame()
    x_df["x"] = pd.Series(np.random.rand(300))
    x_df["a"] = pd.Series(np.random.rand(300))
    x_df["b"] = pd.Series(np.random.rand(300))
    x_df["c"] = pd.Series(np.random.rand(300))
    x_df["d"] = pd.Series(np.random.rand(300))
    x_df.index = pd.date_range(
        start=pd.Timestamp("2020-01-01"),
        periods=300,
        freq='1h'
    )
    coeffs = np.random.randn(5)

    y_df = pd.DataFrame()
    modded = x_df * coeffs

    y_df["x"] = modded.sum(axis=1)
    y_df.index = pd.date_range(
        start=pd.Timestamp("2020-01-03"),
        periods=300,
        freq='1h'
    )

    z_df = pd.DataFrame()
    z_df["x"] = pd.Series(np.random.rand(300))
    z_df["a"] = pd.Series(np.random.rand(300))
    z_df["b"] = pd.Series(np.random.rand(300))
    z_df["c"] = pd.Series(np.random.rand(300))
    z_df["d"] = pd.Series(np.random.rand(300))
    z_df.index = pd.date_range(
        start=pd.Timestamp("2021-01-01"),
        periods=300,
        freq='1h'
    )

    return {"x": x_df.abs(), "y": y_df.abs(), "z": z_df.abs()}


@pytest.mark.parametrize("folds", [2, 3, 4, 5])
@pytest.mark.parametrize("split_prop", [0.1, 0.3, 0.5, 0.9])
@pytest.mark.cal
def test_data_split(full_data, folds, split_prop):
    """
    Tests whether data is split properly
    """
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        folds=folds,
        validation_split=split_prop
    )
    split_coeffs = coeff_inst.return_measurements()["y"]
    result_size = len(split_coeffs)
    tests["Correct Num of Entries"] = result_size == 252
    print(f"Number of entries: {result_size} == 252")
    num_of_folds = split_coeffs.loc[:, "Fold"].nunique()
    test_prop = (
        split_coeffs.loc[:, "Fold"]
        .value_counts()
        .drop("Validation")
        .median() / split_coeffs.shape[0]
    )
    print(split_coeffs.loc[:, "Fold"].value_counts())

    tests["Correct Num of Folds"] = num_of_folds == (folds + 1)
    print(f"Number of folds: {num_of_folds} == {folds+1}")
    fold_prop = (1 - split_prop) / (folds)
    tests["Correct Prop of Folds"] = test_prop == pytest.approx(
         fold_prop, 0.05
    )
    print(f"Fold proportion: {test_prop} ~= {fold_prop}")
    for test, result in tests.items():
        print(f"{test}: {result}")

    assert all(tests.values())


@pytest.mark.parametrize(
    "split_prop",
    [
        (dt.datetime(2020, 1, 5, 0, 0, 0), 204),
        (dt.datetime(2020, 1, 5, 6, 0, 0), 198),
        (dt.datetime(2020, 1, 8, 0, 0, 0), 132),
    ]
)
@pytest.mark.cal
def test_split_by_date(full_data, split_prop):
    """
    Tests whether data is split properly
    """
    folds = 5
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        folds=folds,
        validation_split=split_prop[0]
    )
    split_coeffs = coeff_inst.return_measurements()["y"]
    num_of_folds = split_coeffs.loc[:, "Fold"].nunique()
    result_size = len(split_coeffs)
    tests["Correct Num of Entries"] = result_size == 252
    print(f"Number of entries: {result_size} == 252")

    test_prop = (
        split_coeffs.loc[:, "Fold"].value_counts()[0] / split_coeffs.shape[0]
    )
    print(split_coeffs.loc[:, "Fold"].value_counts())

    tests["Correct Num of Folds"] = num_of_folds == (folds + 1)
    print(f"Number of folds: {num_of_folds} == {folds+1}")
    fold_prop = (1-(split_prop[1]/252)) / (folds)
    tests["Correct Prop of Folds"] = test_prop == pytest.approx(
         fold_prop, 0.05
    )
    print(f"Fold proportion: {test_prop} ~= {fold_prop}")
    for test, result in tests.items():
        print(f"{test}: {result}")

    assert all(tests.values())

@pytest.mark.parametrize(
    "split_prop",
    [
        (dt.timedelta(days=2), 203),
        (dt.timedelta(days=2, hours=6), 197),
        (dt.timedelta(days=5), 131),
    ]
)
@pytest.mark.cal
def test_split_by_delta(full_data, split_prop):
    """
    Tests whether data is split properly
    """
    folds = 5
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        folds=folds,
        validation_split=split_prop[0]
    )
    split_coeffs = coeff_inst.return_measurements()["y"]
    num_of_folds = split_coeffs.loc[:, "Fold"].nunique()
    result_size = len(split_coeffs)
    tests["Correct Num of Entries"] = result_size == 252
    print(f"Number of entries: {result_size} == 252")

    test_prop = (
        split_coeffs.loc[:, "Fold"].value_counts()[0] / split_coeffs.shape[0]
    )
    print(split_coeffs.loc[:, "Fold"].value_counts())

    tests["Correct Num of Folds"] = num_of_folds == (folds + 1)
    print(f"Number of folds: {num_of_folds} == {folds+1}")
    fold_prop = (1-(split_prop[1]/252)) / (folds)
    tests["Correct Prop of Folds"] = test_prop == pytest.approx(
         fold_prop, 0.05
    )
    print(f"Fold proportion: {test_prop} ~= {fold_prop}")
    for test, result in tests.items():
        print(f"{test}: {result}")

    assert all(tests.values())

@pytest.mark.cal()
@pytest.mark.parametrize(
    ("polynomial_degree", "vif_bound"), [(1, None), (2, 5)]
)
@pytest.mark.parametrize(
    "time_col", [False, True]
)
def test_skl_cals(full_data, polynomial_degree, vif_bound, time_col):
    """
    Combines all possible multivariate key combos with each skl calibration
    method except omp which needs at least 1 mv key
    """
    tests = {}
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
        Calibrate.quantile,
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
        Calibrate.linear_gam,
        Calibrate.expectile_gam,
    ]
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        interaction_degree=polynomial_degree,
        vif_bound=vif_bound,
        add_time_column=time_col
    )
    for func in funcs:
        print(func)
        func(coeff_inst)
    models = coeff_inst.return_models()
    tests["Correct number of techniques"] = len(models.keys()) == len(funcs)
    for technique, scaling_methods in models.items():
        if vif_bound:
            continue
        tests[f"Correct number of scalers {technique}"] = (
            len(scaling_methods.keys()) == 1
        )
        for _, var_combos in scaling_methods.items():
            min_coeffs_2 = (
                "Orthogonal Matching Pursuit" in technique
            )
            if min_coeffs_2:
                correct_num_of_vars = 1
            elif "Isotonic Regression" in technique:
                correct_num_of_vars = 1
            else:
                correct_num_of_vars = 2
            if time_col and "Isotonic Regression" not in technique:
                correct_num_of_vars = correct_num_of_vars + 2

            print(
                f"{technique}: {len(var_combos.keys())}"
                f"== {correct_num_of_vars}"
            )
            tests[f"Correct number of vars {technique}"] = (
                len(var_combos.keys()) == correct_num_of_vars
            )
            for vars, folds in var_combos.items():
                print(technique, vars, len(folds.keys()))
                if (
                    # "Cross Validated" not in technique and
                    "(Random Search)"
                    not in technique
                ):
                    tests[f"Correct number of folds {technique} {vars}"] = (
                        len(folds.keys()) == 5
                    )
                else:
                    tests[f"Correct number of folds {technique} {vars}"] = (
                        len(folds.keys()) == 1
                    )

                for fold, pipe in folds.items():
                    print(pipe[:-1].get_feature_names_out())
                    _ = pipe.predict(full_data["z"])
                    tests[
                        f"Pipe for {technique} {vars} {fold} works"
                    ] = isinstance(pipe, Pipeline)

    for test, result in tests.items():
        if not result:
            print(f"{test}: {result}")
    assert all(tests.values())


def test_skl_cals_random_search(full_data):
    """
    Combines all possible multivariate key combos with each skl calibration
    method except omp which needs at least 1 mv key
    """
    tests = {}
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
        Calibrate.quantile,
        Calibrate.random_forest,
        Calibrate.ridge,
        Calibrate.ridge_cv,
        Calibrate.stochastic_gradient_descent,
        Calibrate.svr,
        Calibrate.theil_sen,
        Calibrate.tweedie,
        Calibrate.xgboost,
        Calibrate.xgboost_rf,
    ]
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        random_search_iterations=3
    )
    for func in funcs:
        print(func)
        func(coeff_inst, random_search=True)
    models = coeff_inst.return_models()
    tests["Correct number of techniques"] = len(models.keys()) == len(funcs)
    for technique, scaling_methods in models.items():
        tests[f"Correct number of scalers {technique}"] = (
            len(scaling_methods.keys()) == 1
        )
        for _, var_combos in scaling_methods.items():
            min_coeffs_2 = (
                "Orthogonal Matching Pursuit" in technique
            )
            if min_coeffs_2:
                correct_num_of_vars = 1
            elif "Isotonic Regression" in technique:
                correct_num_of_vars = 1
            else:
                correct_num_of_vars = 2

            print(
                f"{technique}: {len(var_combos.keys())}"
                f"== {correct_num_of_vars}"
            )
            tests[f"Correct number of vars {technique}"] = (
                len(var_combos.keys()) == correct_num_of_vars
            )
            for vars, folds in var_combos.items():
                print(technique, vars, len(folds.keys()))
                tests[f"Correct number of folds {technique} {vars}"] = (
                    len(folds.keys()) == 5
                )

                for fold, pipe in folds.items():
                    print(pipe[:-1].get_feature_names_out())
                    _ = pipe.predict(full_data["z"])
                    tests[
                        f"Pipe for {technique} {vars} {fold} works"
                    ] = isinstance(pipe, Pipeline)

    for test, result in tests.items():
        if not result:
            print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
@pytest.mark.parametrize(
    "proportion", [None, 0.9, 200]
)
def test_subsample(full_data, proportion):
    """Test setting subsample of data."""
    tests = {}
    df_size = 252
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        subsample_data=proportion,
    )
    measurements = coeff_inst.return_measurements()
    tests['Same size'] = (
            measurements['x'].shape[0] == measurements['y'].shape[0]
    )
    if proportion is None:
        tests['Correct size'] = df_size == measurements['x'].shape[0]
    elif isinstance(proportion, int):
        tests['Correct size'] = proportion == measurements['x'].shape[0]
        print(f"Correct size: {proportion} == {measurements['x'].shape[0]}")
    else:
        tests['Correct size'] = df_size * proportion == pytest.approx(
                    measurements['x'].shape[0], 3
        )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_subsample_failsafe(full_data):
    """Test setting subsample of data."""
    tests = {}
    df_size = 252
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        subsample_data=10000,
    )
    measurements = coeff_inst.return_measurements()

    tests['Same size'] = (
            measurements['x'].shape[0] == measurements['y'].shape[0]
    )
    tests['Correct size'] = measurements['x'].shape[0] == df_size

    for test, result in tests.items():
        if not result:
            print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_vif_correct_col_names(full_data):
    tests = {}
    full_data["x"]["e"] = full_data["x"]["d"].mul(1.5)
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        vif_bound=3,
    )
    coeff_inst.linreg()
    models = coeff_inst.return_models()["Linear Regression"]["None"]
    tests["Single variable present"] = "x" in models
    tests["All variables present"] = (
            "x + a + b + c + d + e" in models
    )
    tests["Two variable combinations present"] = (
            len(models) == 2
    )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())

@pytest.mark.cal()
def test_only_one_feature(full_data):
    tests = {}
    full_data["x"] = full_data["x"][["x"]]
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        vif_bound=3,
    )
    coeff_inst.linreg()
    models = coeff_inst.return_models()["Linear Regression"]["None"]
    tests["Single variable present"] = "x" in models
    tests["Two variable combinations present"] = (
            len(models) == 1
    )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_save_as_pickle(tmpdir, full_data):
    temp_path = Path(tmpdir) / "pkl"
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        pickle_path=temp_path,
        target="x",
        vif_bound=3,
    )
    coeff_inst.linreg()
    coeff_inst.decision_tree()
    coeff_inst.xgboost()
    models = coeff_inst.return_models()
    tests["Three calibration models"] = (
            len(models) == 3
    )
    print(models.keys())
    tests["Correct calibration models"] = (
        set(models.keys()) == {
            "Linear Regression",
            "Decision Tree",
            "XGBoost Regression",
        }
    )
    for cal_model, vars in models.items():
        tests[f"{cal_model}: 2 var combos present"] = (
            len(vars["None"]) == 2
        )
        for var, folds in vars["None"].items():
            tests[f"{cal_model}, {var}: 5 folds present"] = (
                len(folds) == 5
            )
            tests[f"{cal_model}, {var}: Correct fold names"] = (
                set(folds.keys()) == {0,1,2,3,4}
            )
            for fold, pkl_path in folds.items():
                tests[f"{cal_model}, {var}, {fold}: Is path"] = (
                    isinstance(pkl_path, Path)
                )
                tests[f"{cal_model}, {var}, {fold}: Exists"] = (
                    pkl_path.exists()
                )
                with pkl_path.open("rb") as file:
                    pkl = pickle.load(file)
                tests[f"{cal_model}, {var}, {fold}: Is pickled pipeline"] = (
                    isinstance(pkl, Pipeline)
                )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_bad_val_split(full_data):
    with pytest.raises(
        TypeError,
        match="Expected float, datetime or timedelta for validation split"
    ):
        _ = Calibrate.setup(
            x_data=full_data["x"],
            y_data=full_data["y"],
            target="x",
            validation_split="BREAK IT"
        )


@pytest.mark.cal()
@pytest.mark.parametrize("scaling_alg", ["Not a scaling algorithm", []])
def test_bad_scaler(full_data, scaling_alg):
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        scaler=scaling_alg
    )
    coeff_inst.linreg()
    coeff_inst.decision_tree()
    coeff_inst.xgboost()
    models = coeff_inst.return_models()
    tests["Three calibration models"] = (
            len(models) == 3
    )
    print(models.keys())
    tests["Correct calibration models"] = (
        set(models.keys()) == {
            "Linear Regression",
            "Decision Tree",
            "XGBoost Regression",
        }
    )
    for cal_model, scalers in models.items():
        tests[f"{cal_model}: 7 scalers present"] = (
            len(scalers) == 1
        )
        tests[f"{cal_model}: All scalers present"] = (
            set(scalers.keys()) == {
                "None",
            }
        )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_clear_models(full_data):
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        scaler='All'
    )
    coeff_inst.linreg()
    coeff_inst.decision_tree()
    coeff_inst.xgboost()
    coeff_inst.clear_models()
    models = coeff_inst.return_models()
    tests["No calibration models"] = (
            len(models) == 0
    )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_all_scalers_pos(full_data):
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        scaler='All'
    )
    coeff_inst.linreg()
    coeff_inst.decision_tree()
    coeff_inst.xgboost()
    models = coeff_inst.return_models()
    tests["Three calibration models"] = (
            len(models) == 3
    )
    print(models.keys())
    tests["Correct calibration models"] = (
        set(models.keys()) == {
            "Linear Regression",
            "Decision Tree",
            "XGBoost Regression",
        }
    )
    for cal_model, scalers in models.items():
        tests[f"{cal_model}: 7 scalers present"] = (
            len(scalers) == 7
        )
        tests[f"{cal_model}: All scalers present"] = (
            set(scalers.keys()) == {
                "None",
                "Standard Scale",
                "MinMax Scale",
                "Yeo-Johnson Transform",
                "Box-Cox Transform",
                "Quantile Transform (Uniform)",
                "Quantile Transform (Gaussian)"
            }
        )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_all_scalers_neg(full_data):
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"].mul(-1),
        y_data=full_data["y"],
        target="x",
        scaler='All'
    )
    coeff_inst.linreg()
    coeff_inst.decision_tree()
    coeff_inst.xgboost()
    models = coeff_inst.return_models()
    tests["Three calibration models"] = (
            len(models) == 3
    )
    print(models.keys())
    tests["Correct calibration models"] = (
        set(models.keys()) == {
            "Linear Regression",
            "Decision Tree",
            "XGBoost Regression",
        }
    )
    for cal_model, scalers in models.items():
        tests[f"{cal_model}: 6 scalers present"] = (
            len(scalers) == 6
        )
        tests[f"{cal_model}: All scalers present except BC"] = (
            set(scalers.keys()) == {
                "None",
                "Standard Scale",
                "MinMax Scale",
                "Yeo-Johnson Transform",
                "Quantile Transform (Uniform)",
                "Quantile Transform (Gaussian)"
            }
        )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_boxcox_scalers_neg(full_data):
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"].mul(-1),
        y_data=full_data["y"],
        target="x",
        scaler=["None", "Box-Cox Transform"]
    )
    coeff_inst.linreg()
    coeff_inst.decision_tree()
    coeff_inst.xgboost()
    models = coeff_inst.return_models()
    tests["Three calibration models"] = (
            len(models) == 3
    )
    print(models.keys())
    tests["Correct calibration models"] = (
        set(models.keys()) == {
            "Linear Regression",
            "Decision Tree",
            "XGBoost Regression",
        }
    )
    for cal_model, scalers in models.items():
        tests[f"{cal_model}: 1 scaler present"] = (
            len(scalers) == 1
        )
        tests[f"{cal_model}: All scalers present"] = (
            set(scalers.keys()) == {
                "None"
            }
        )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_bad_scaler_in_list(full_data):
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        scaler=["None", "This scaler does not exist"]
    )
    coeff_inst.linreg()
    coeff_inst.decision_tree()
    coeff_inst.xgboost()
    models = coeff_inst.return_models()
    tests["Three calibration models"] = (
            len(models) == 3
    )
    print(models.keys())
    tests["Correct calibration models"] = (
        set(models.keys()) == {
            "Linear Regression",
            "Decision Tree",
            "XGBoost Regression",
        }
    )
    for cal_model, scalers in models.items():
        tests[f"{cal_model}: 1 scaler present"] = (
            len(scalers) == 1
        )
        tests[f"{cal_model}: All scalers present"] = (
            set(scalers.keys()) == {
                "None"
            }
        )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_bad_scaler_type(full_data):
    with pytest.raises(
        TypeError,
        match="Scaler parameter should be string, list or tuple"
    ):
        _ = Calibrate.setup(
            x_data=full_data["x"],
            y_data=full_data["y"],
            target="x",
            scaler=49
        )

@pytest.mark.cal()
def test_boxcox_scalers_neg(full_data):
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"].mul(-1),
        y_data=full_data["y"],
        target="x",
        scaler=["None", "Box-Cox Transform"]
    )
    coeff_inst.linreg()
    coeff_inst.decision_tree()
    coeff_inst.xgboost()
    models = coeff_inst.return_models()
    tests["Three calibration models"] = (
            len(models) == 3
    )
    print(models.keys())
    tests["Correct calibration models"] = (
        set(models.keys()) == {
            "Linear Regression",
            "Decision Tree",
            "XGBoost Regression",
        }
    )
    for cal_model, scalers in models.items():
        tests[f"{cal_model}: 1 scaler present"] = (
            len(scalers) == 1
        )
        tests[f"{cal_model}: All scalers present"] = (
            set(scalers.keys()) == {
                "None"
            }
        )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_time_col_correct_col_names(full_data):
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        add_time_column=True
    )
    coeff_inst.linreg()
    models = coeff_inst.return_models()["Linear Regression"]["None"]
    tests["Single variable present"] = "x" in models
    tests["Single variable + Time col present"] = (
            "x + Time Since Origin" in models
    )
    tests["All variables present"] = (
            "x + a + b + c + d" in models
    )
    tests["All variables + Time col present"] = (
            "x + a + b + c + d + Time Since Origin" in models
    )
    tests["Four variable combinations present"] = (
            len(models) == 4
    )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_poly_feat_correct_col_names(full_data):
    tests = {}
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        interaction_degree=2,
        interaction_features=["b", "c", "d"]
    )
    coeff_inst.linreg()
    models = coeff_inst.return_models()["Linear Regression"]["None"]
    print(models.keys())
    tests["Single variable present"] = "x" in models
    tests["All variables present"] = (
            "x + a + b + c + d + b^2 + b c + b d + c^2 + c d + d^2" in models
    )
    tests["Two variable combinations present"] = (
            len(models) == 2
    )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_vif_removes_all(full_data):
    tests = {}
    for col in ["a", "b", "c", "d"]:
        full_data["x"][col] = full_data["x"]["x"]
    coeff_inst = Calibrate.setup(
        x_data=full_data["x"],
        y_data=full_data["y"],
        target="x",
        vif_bound=1
    )
    coeff_inst.linreg()
    models = coeff_inst.return_models()["Linear Regression"]["None"]
    print(models.keys())
    tests["Single variable present"] = "x" in models
    tests["All variables present"] = (
            "x + a + b + c + d" in models
    )
    tests["Two variable combinations present"] = (
            len(models) == 2
    )
    filtered_features = models["x"][0][-2].get_filtered_features()
    tests["All features filtered except one"] = (
        len(filtered_features) == 1
    )
    tests["Target is filtered feature"] = (
        'x' in filtered_features
    )
    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.cal()
def test_time_col_trans_error_transform_not_fitted():
    tct = TimeColumnTransformer()
    with pytest.raises(ValueError, match="Transformer not fitted"):
        tct.transform(None, None)


@pytest.mark.cal()
def test_time_col_trans_error_featrues_not_fitted():
    tct = TimeColumnTransformer()
    with pytest.raises(ValueError, match="Transformer not fitted"):
        tct.get_feature_names_out()


@pytest.mark.cal()
def test_target_not_in_both_cols_error(full_data):
    with pytest.raises(ValueError, match="x does not exist in both columns"):
        _ = Calibrate.setup(
            x_data=full_data["x"].drop(["x"], axis=1),
            y_data=full_data["y"],
            target="x"
        )

