from io import StringIO

import pandas as pd
import pytest

from calidhayte import Calibrate


@pytest.fixture
def all_values_present_skl():
    """
    Coeffs and measurements with no expected errors
    """
    coeffs = """Coefficients,coeff.x,coeff.a,coeff.b,i.intercept
x,2,,,1
a,,2,,1
x+a,1,1,,1
b,,,4,1
x+b,0.5,,3,1
x+a+b,0.25,0.25,3,1"""
    test = """index,y,x,a,b
0,3,1,1,0.5
1,5,2,2,1
3,17,8,8,4"""
    train = """index,y,x,a,b
2,9,4,4,2
4,1,0,0,0
    """
    return {
            'train': pd.read_csv(StringIO(train)).set_index("index"),
            'test': pd.read_csv(StringIO(test)).set_index("index"),
            'coefficients': pd.read_csv(StringIO(coeffs)).set_index(
                "Coefficients"
                )
            }


@pytest.fixture
def all_values_present_pymc():
    """
    Coeffs and measures with no expected errors
    """

    coeffs = """Coefficients,coeff.x,sd.x,coeff.a,sd.a,i.intercept,sd.intercept
x,2,1,,,3,1
a,,,2,1,3,1
x+a,1,1,1,1,3,1"""

    test = """index,y,x,a
0,5,1,1
1,7,2,2
3,4,0.5,0.5"""

    train = """index,y,x,a
2,11,4,4
4,3,0,0"""

    return {
            'train': pd.read_csv(StringIO(train)).set_index("index"),
            'test': pd.read_csv(StringIO(test)).set_index("index"),
            'coefficients': pd.read_csv(StringIO(coeffs)).set_index(
                "Coefficients"
                )
            }


def test_skl_standard_cal(all_values_present_skl):
    """
    Tests that signals with associated scikitlearn coefficients are calibrated
    properly
    """
    tests = dict()
    cal = Calibrate(**all_values_present_skl)

    expected_test = all_values_present_skl['test'].loc[:, "y"]
    expected_train = all_values_present_skl['train'].loc[:, "y"]

    measures = cal.return_measurements()
    for col in measures["Train"].columns:
        tests[f"{col} train"] = expected_train.equals(measures["Train"]
                                                      .loc[:, col]
                                                      .astype(int)
                                                      )
    for col in measures["Test"].columns:
        tests[f"{col} test"] = expected_test.equals(measures["Test"]
                                                    .loc[:, col]
                                                    .astype(int)
                                                    )
    for key, test in tests.items():
        print(f"{key}: {test}")
    assert all(tests.values())


def test_pymc_standard_cal(all_values_present_pymc):
    """
    Tests that signals with associated pymc coefficients are calibrated
    properly
    """
    tests = dict()
    cal = Calibrate(**all_values_present_pymc)

    expected_test = all_values_present_pymc['test'].loc[:, "y"]
    expected_train = all_values_present_pymc['train'].loc[:, "y"]

    keys = ["x", "a", "x+a"]
    vals = cal.return_measurements()
    for key in keys:
        tests[f"{key} train"] = expected_train.equals(vals['mean.Train'][key]
                                                      .astype(int)
                                                      )
        tests[f"{key} min train"] = expected_train.gt(vals['min.Train'][key]
                                                      .astype(float)
                                                      ).all()
        tests[f"{key} max train"] = expected_train.lt(vals['max.Train'][key]
                                                      .astype(float)
                                                      ).all()

        tests[f"{key} test"] = expected_test.equals(vals['mean.Test'][key]
                                                    .astype(int)
                                                    )
        tests[f"{key} min test"] = expected_test.gt(vals['min.Test'][key]
                                                    .astype(float)
                                                    ).all()
        tests[f"{key} max test"] = expected_test.lt(vals['max.Test'][key]
                                                    .astype(float)
                                                    ).all()
    for key, test in tests.items():
        print(f"{key}: {test}")
    assert all(tests.values())


def test_calibrate_blanks_provided():
    """
    Tests that the proper error is raised when passing in blank dataframes
    """
    with pytest.raises(
            ValueError,
            match=r"The following axis are empty: \[.*\]"
            ):
        Calibrate(
                train=pd.DataFrame(),
                test=pd.DataFrame(),
                coefficients=pd.DataFrame()
                )


@pytest.mark.parametrize(
        "data_type,ex_or,ex_pr",
        [
            (
                "skl",
                pd.Series([1, 2, 4, 8, 0]),
                pd.Series([3, 5, 9, 17, 1])
            ),
            (
                "pymc",
                pd.Series([1, 2, 4, 0.5, 0]),
                pd.Series([5, 7, 11, 4, 3])
            )
         ]
    )
def test_join_measurements(
        data_type,
        ex_or,
        ex_pr,
        all_values_present_skl,
        all_values_present_pymc
        ):
    """
    Tests that measurements are joined properly
    """
    tests = list()
    if data_type == "skl":
        measures = all_values_present_skl
    else:
        measures = all_values_present_pymc
    cal = Calibrate(**measures)

    joined_measures = cal.join_measurements()

    orig_test = ex_or.eq(joined_measures["x"].loc[:, "x"])
    tests.append(orig_test.all())
    print(f"x values joined correctly: {orig_test}")
    pred_test = ex_pr.eq(joined_measures["y"].loc[:, "x"])
    tests.append(pred_test.all())
    print(f"y values joined correctly: {pred_test}")

    assert all(tests)
