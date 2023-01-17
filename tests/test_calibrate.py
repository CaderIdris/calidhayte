from io import StringIO

import pandas as pd
import pytest

from calidhayte import Calibrate


@pytest.fixture
def all_values_present_skl():
    """
    Coeffs and measurements with no expected errors
    """
    coeffs = """Coefficients,coeff.x,coeff.a,coeff.b,i.Intercept
    x,2,,,1
    a,0,2,,1
    x+a,1,1,,1
    b,0,,4,1
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
