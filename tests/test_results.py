from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calidhayte.calibrate import Calibrate
from calidhayte.results import (
    Results
)


@pytest.fixture
def trained_models():
    """ """
    np.random.seed(4)
    x_df = pd.DataFrame()
    x_df["x"] = pd.Series(np.random.rand(300))
    x_df["a"] = pd.Series(np.random.rand(300))
    x_df["b"] = pd.Series(np.random.rand(300))
    x_df["c"] = pd.Series(np.random.rand(300))
    coeffs = np.random.randn(4)

    y_df = pd.DataFrame()
    modded = x_df * coeffs

    y_df["x"] = modded.sum(axis=1).abs().add(1)
    y_df["Fold"] = (
        ([0] * 50)
        + ([1] * 50)
        + ([2] * 50)
        + ([3] * 50)
        + ([4] * 50)
        + (["Validation"] * 50)
    )

    cal = Calibrate.setup(x_df, y_df, target="x")
    cal.linreg()
    cal.theil_sen()
    cal.random_forest()

    return {
        "x_data": x_df.abs(),
        "y_data": y_df,
        "target": "x",
        "models": cal.return_models(),
    }


@pytest.fixture
def trained_models_pkl(tmpdir):
    """ """
    pkl_path = Path(tmpdir) / 'pkl_results'
    np.random.seed(4)
    x_df = pd.DataFrame()
    x_df["x"] = pd.Series(np.random.rand(300))
    x_df["a"] = pd.Series(np.random.rand(300))
    x_df["b"] = pd.Series(np.random.rand(300))
    x_df["c"] = pd.Series(np.random.rand(300))
    coeffs = np.random.randn(4)

    y_df = pd.DataFrame()
    modded = x_df * coeffs

    y_df["x"] = modded.sum(axis=1).abs().add(1)
    y_df["Fold"] = (
        ([0] * 50)
        + ([1] * 50)
        + ([2] * 50)
        + ([3] * 50)
        + ([4] * 50)
        + (["Validation"] * 50)
    )

    cal = Calibrate.setup(
        x_df,
        y_df,
        target="x",
        pickle_path=pkl_path
    )
    cal.linreg()
    cal.theil_sen()
    cal.random_forest()

    return {
        "x_data": x_df.abs(),
        "y_data": y_df,
        "target": "x",
        "models": cal.return_models(),
    }


@pytest.mark.results
def test_prepare_datasets(trained_models):
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
        Results.root_mean_squared_log,
        Results.median_absolute,
        Results.mean_absolute_percentage,
        Results.r2,
        Results.mean_pinball_loss,
        Results.mean_poisson_deviance,
        Results.mean_gamma_deviance,
        Results.mean_tweedie_deviance,
        Results.centered_rmse,
        Results.mbe,
        Results.ref_iqr,
        Results.ref_mean,
        Results.ref_range,
        Results.ref_sd,
        Results.ref_mad,
        Results.unbiased_rmse,
    ]
    for err in errs:
        err(results)

    res = results.return_errors()
    tests["Correct num of techniques"] = res.nunique()["Technique"] == 4
    tests["Correct num of vars"] = res.nunique()["Variables"] == 2
    tests["Correct num of folds"] = res.nunique()["Fold"] == 6

    tests["Correct num of rows"] = res.shape[0] == (3 * 2 * 6) + 1

    tests["Correct num of cols"] = res.shape[1] == 25

    assert all(tests.values())


@pytest.mark.results
def test_prepare_datasets_pkl(trained_models_pkl):
    """
    Tests whether all datasets are selected properly
    """
    tests = dict()

    results = Results(**trained_models_pkl)

    errs = [
        Results.explained_variance_score,
        Results.max,
        Results.mean_absolute,
        Results.root_mean_squared,
        Results.root_mean_squared_log,
        Results.median_absolute,
        Results.mean_absolute_percentage,
        Results.r2,
        Results.mean_pinball_loss,
        Results.mean_poisson_deviance,
        Results.mean_gamma_deviance,
        Results.mean_tweedie_deviance,
        Results.centered_rmse,
        Results.mbe,
        Results.ref_iqr,
        Results.ref_mean,
        Results.ref_range,
        Results.ref_sd,
        Results.ref_mad,
        Results.unbiased_rmse,
    ]
    for err in errs:
        err(results)

    res = results.return_errors()
    tests["Correct num of techniques"] = res.nunique()["Technique"] == 4
    tests["Correct num of vars"] = res.nunique()["Variables"] == 2
    tests["Correct num of folds"] = res.nunique()["Fold"] == 6

    tests["Correct num of rows"] = res.shape[0] == (3 * 2 * 6) + 1

    tests["Correct num of cols"] = res.shape[1] == 25

    assert all(tests.values())


@pytest.mark.results()
def test_squared_cmse(trained_models):
    tests = {}

    results = Results(**trained_models, errors=pd.DataFrame())
    results.centered_rmse(squared=True)

    res = results.return_errors()
    tests["Correct num of techniques"] = res.nunique()["Technique"] == 4
    tests["Correct num of vars"] = res.nunique()["Variables"] == 2
    tests["Correct num of folds"] = res.nunique()["Fold"] == 6

    tests["Correct num of rows"] = res.shape[0] == (3 * 2 * 6) + 1
    print(res.to_csv())
    print(res.shape)

    tests["Correct num of cols"] = res.shape[1] == 6

    assert all(tests.values())


@pytest.mark.results()
def test_bad_target(trained_models):
    trained_models["target"] = "Bad target"
    with pytest.raises(
        ValueError,
        match="Bad target not in x and y"
    ):
        _ = Results(**trained_models, errors=pd.DataFrame())


@pytest.mark.results()
def test_neg_data_rmsle(trained_models):
    trained_models["y_data"]["x"] = trained_models["y_data"]["x"].mul(-1)
    with pytest.raises(
        ValueError
    ):
        results = Results(**trained_models, errors=pd.DataFrame())
        results.root_mean_squared_log()
