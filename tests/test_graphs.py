import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from calidhayte.calibrate import Calibrate
from calidhayte.graphs import Graphs


@pytest.fixture
def trained_models():
    """
    """
    np.random.seed(4)
    x, y = make_regression(n_samples=300, n_features=4, n_targets=1)
    x_df = pd.DataFrame(data=x, columns=['x', 'a', 'b', 'c'])

    y_df = pd.DataFrame()
    y_df['x'] = y
    y_df['Fold'] = ([0]*60) + ([1]*60) + ([2]*60) + ([3]*60) + ([4]*60)

    cal = Calibrate(
            x_df,
            y_df,
            target='x',
            scaler='All'
            )
    cal.linreg()
    cal.theil_sen()
    cal.random_forest()

    return {
            'x': x_df,
            'y': y_df,
            'target': 'x',
            'models': cal.return_models(),
            'x_name': 'x',
            'y_name': 'y'
            }


@pytest.mark.plots
def test_linreg(
        trained_models
        ):
    """
    Tests whether all datasets are selected properly
    """
    print(trained_models['x'])
    print(trained_models['y'])

    results = Graphs(**trained_models)

    results.lin_reg_plot()
    results.save_plots('.tmp/tests')


@pytest.mark.plots
def test_ecdf(
        trained_models
        ):
    """
    Tests whether all datasets are selected properly
    """
    print(trained_models['x'])
    print(trained_models['y'])

    results = Graphs(**trained_models)

    results.ecdf_plot()
    results.save_plots('.tmp/tests')


@pytest.mark.plots
def test_bland_altman(
        trained_models
        ):
    """
    Tests whether all datasets are selected properly
    """
    print(trained_models['x'])
    print(trained_models['y'])

    results = Graphs(**trained_models)

    results.bland_altman_plot()
    results.save_plots('.tmp/tests')
