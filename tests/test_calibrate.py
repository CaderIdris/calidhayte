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
    tests = dict()
    cal = Calibrate(**all_values_present_skl)

    expected_test = all_values_present_skl['test'].loc[:, "y"]
    expected_train = all_values_present_skl['train'].loc[:, "y"]

    for key, pred in cal.return_measurements().items():
        tests[f"{key} train"] = expected_train.equals(pred['Train']
                                                      .astype(int)
                                                      )
        tests[f"{key} test"] = expected_test.equals(pred['Test']
                                                    .astype(int)
                                                    )
    for key, test in tests.items():
        print(f"{key}: {test}")
    assert all(tests.values())


def test_pymc_standard_cal(all_values_present_pymc):
    tests = dict()
    cal = Calibrate(**all_values_present_pymc)

    expected_test = all_values_present_pymc['test'].loc[:, "y"]
    expected_train = all_values_present_pymc['train'].loc[:, "y"]

    keys = ["x", "a", "x+a"]
    vals = cal.return_measurements()

    for key in keys:
        tests[f"{key} train"] = expected_train.equals(vals[key]['mean.Train']
                                                      .astype(int)
                                                      )
        tests[f"{key} min train"] = expected_train.gt(vals[key]['min.Train']
                                                      .astype(float)
                                                      ).all()
        tests[f"{key} max train"] = expected_train.lt(vals[key]['max.Train']
                                                      .astype(float)
                                                      ).all()

        tests[f"{key} test"] = expected_test.equals(vals[key]['mean.Test']
                                                    .astype(int)
                                                    )
        tests[f"{key} min test"] = expected_test.gt(vals[key]['min.Test']
                                                    .astype(float)
                                                    ).all()
        tests[f"{key} max test"] = expected_test.lt(vals[key]['max.Test']
                                                    .astype(float)
                                                    ).all()
    for key, test in tests.items():
        print(f"{key}: {test}")
    assert all(tests.values())


def test_calibrate_blanks_provided():
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
        "measures,ex_or,ex_pr",
        [
            (
                all_values_present_skl,
                pd.Series(1, 2, 3, 8, 0),
                pd.Series(3, 5, 9, 17, 1)
            ),
            (
                all_values_present_pymc,
                pd.Series(1, 2, 4, 0.5, 0),
                pd.Series(5, 7, 11, 4, 3)
            )
         ]
    )
def test_join_measurements(all_values_present_skl):
    tests = list()
    cal = Calibrate(**all_values_present_skl)

    joined_measures = cal.join_measurements()
    expected_original = pd.Series([1, 2, 4, 8, 0])
    expected_pred = pd.Series([3, 5, 9, 17, 1])

    orig_test = expected_original.eq(joined_measures.get("x").loc[:, "x"])
    tests.append(orig_test)
    print(f"x values joined correctly: {orig_test}")
    pred_test = expected_pred.eq(joined_measures.get("y").loc[:, "x"])
    tests.append(pred_test)
    print(f"y values joined correctly: {pred_test}")

    assert all(tests)
