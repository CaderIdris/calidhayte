from calidhayte import Summary
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def test_dicts() -> dict[str, pd.DataFrame]:
    obvious_winner = pd.DataFrame()
    dict_vals = {
            'a+a': 0.1,
            'b+b+b': 0.2,
            'c+c+c+c': 0.3,
            'd+d+d+d+d': 0.4
            }
    for col in ["t1", "t2", "t3", "t4"]:
        obvious_winner[col] = dict_vals

    second_place = obvious_winner + 1
    obvious_loser = obvious_winner + 2

    return {
            'Obvious Winner': obvious_winner,
            'Second Place': second_place,
            'Obvious Loser': obvious_loser
            }


@pytest.mark.summary
def test_min(test_dicts):
    sum = Summary(test_dicts)
    assert sum.min().equals(test_dicts['Obvious Winner'])


@pytest.mark.summary
def test_max(test_dicts):
    sum = Summary(test_dicts)
    assert sum.max().equals(test_dicts['Obvious Loser'])


@pytest.mark.summary
def test_mean(test_dicts):
    sum = Summary(test_dicts)
    assert np.isclose(sum.mean(), test_dicts['Second Place']).all()


@pytest.mark.summary
def test_median(test_dicts):
    sum = Summary(test_dicts)
    assert sum.median().equals(test_dicts['Second Place'])


@pytest.mark.summary
def test_diff_from_mean(test_dicts):
    tests = dict()
    sum = Summary(test_dicts)
    diffs = sum.diff_from_mean()

    tests['Obvious Winner'] = np.isclose(diffs['Obvious Winner'], -1).all()
    tests['Second Place'] = np.isclose(diffs['Second Place'], 0).all()
    tests['Obvious Loser'] = np.isclose(diffs['Obvious Loser'], 1).all()

    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())


@pytest.mark.summary
def test_best_performing(test_dicts):
    tests = dict()
    sum = Summary(test_dicts)
    best_all = sum.best_performing(summate="all")
    best_key = sum.best_performing(summate="key")
    best_row = sum.best_performing(summate="row")

    # Test all
    tests['Summating all'] = all(
            [
                (pd.Series(best_all['Obvious Winner']) == 4).all(),
                (pd.Series(best_all['Second Place']) == 0).all(),
                (pd.Series(best_all['Obvious Loser']) == 0).all()
                ]
            )

    # Test key
    tests['Summating key'] = all(
            [
                best_key['Obvious Winner'] == 16,
                best_key['Second Place'] == 0,
                best_key['Obvious Loser'] == 0,
                ]
            )

    tests['Summating row'] = (pd.Series(best_row) == 4).all()

    for test, result in tests.items():
        print(f"{test}: {result}")
    assert all(tests.values())
