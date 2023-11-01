import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from calidhayte.calibrate import Calibrate
from calidhayte.results import Results
from calidhayte.summary import Summary


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
            target='x'
            )
    cal.linreg()
    cal.theil_sen()
    cal.random_forest()

    models = cal.return_models()

    results = Results(
            x_df,
            y_df,
            'x',
            models
            )

    results.max()
    results.mean_absolute()
    results.root_mean_squared()
    results.r2()
    results.mean_absolute_percentage()

    res = results.return_errors()

    return {
            'results': res,
            'cols': [
                'Max Error',
                'r2',
                'Root Mean Squared Error',
                'Mean Absolute Error',
                'Mean Absolute Percentage Error'
                ]
            }


@pytest.mark.summary
def test_boxplots(
        trained_models
        ):
    """
    Tests whether all datasets are selected properly
    """
    results = Summary(**trained_models)
    results.boxplots()
    results.save_plots('.tmp/tests')


@pytest.mark.summary
def test_histograms(
        trained_models
        ):
    """
    Tests whether all datasets are selected properly
    """
    results = Summary(**trained_models)
    results.histograms()
    results.save_plots('.tmp/tests')
