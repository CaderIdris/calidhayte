<h1 align="center">
	calidhayte
</h1>

**Contact**: [CaderIdrisGH@outlook.com](mailto:CaderIdrisGH@outlook.com)

![Tests](https://github.com/CaderIdris/calidhayte/actions/workflows/tests.yml/badge.svg)
 
---

## Table of Contents

1. [Summary](##summary)
1. [Main Features](##main-features)
1. [How to Install](##how-to-install)
1. [Dependencies](##dependencies)
1. [Example Usage](##example-usage)
1. [Acknowledgements](##acknowledgements)

---

## Summary

calidhayte contains a series of modules designed to calibrate one set of measurements against a reference/true value, perform a suite of error calculations on the results and plot several different types of graphs to summarise the results.
The measurements are split into test and training set and the tests can be performed on a configurable subset of measurements:
- Uncalibrated Test/Train/Full
- Calibrated Test/Train/Full

This module was designed with air quality measurements in mind though other measurements should work too.

---

## Main Features

- Calibrate one set of measurements (with a configurable number of secondary independent variables) against a reference or true value
	- A suite of calibration methods are available, including bayesian methods
- Perform a suite of error calculations on the resulting calibration
---

## How to install

NOTE: Bayesian regression via pymc currently does not work for Python 3.11.
Therefore Python 3.11 is currently not supported for Bayesian methods

**pip**

```bash
pip install git+https://github.com/CaderIdris/calidhayte@{release_tag}
```

**conda**
```bash
conda install git pip
pip install git+https://github.com/CaderIdris/calidhayte@{release_tag} 
```

The release tags can be found in the sidebar

---

## Dependencies

Please see [requirements.txt](./requirements.txt) and [requirements_dev.txt](./requirements_dev.txt) for the standard and development dependencies respectively.

---

## Example Usage

This module requires two dataframes as a prerequisite. 

**Independent Measurements**
||Values|a|b|c|d|e|
|---|---|---|---|---|---|---|
|2022-01-01|0.1|0|7|2.2|3|5|
|2022-01-02|0.7|1|3|2|8.9|1|
|2022-01-03|*nan*|*nan*|1|*nan*|*nan*|7|
|_|_|_|_|_|_|_|
|2022-09-30|0.5|3|1|2.7|4|0|

**Dependent Measurements**
||Values|
|---|---|
|2022-01-02|1|
|2022-01-05|3|
|_|_|
|2022-09-29|*nan*|
|2022-09-30|37|
|2022-10-01|3|

- The two dataframes are joined on the index as an inner join, so the indices do not have to match initially
- *nan* values can be present
- More than one column can be present for the dependent measurements but only 'Values' will be used
- The index can contain date objects, datetime objects or integers. They should be unique. Strings are untested and may cause unexpected behaviours


```python
from calidhayte import Coefficients, Results, Graphs, Summary

# x_df is a dataframe containing multiple columns containing independent measurements.
# The primary measurement is denoted by the 'Values' columns, the other measurement columns can have any name.
# y_df is a dataframe containing the dependent measurement in the 'Values' column.

coeffs = Coefficients(
	x_data=x_df,
	y_data=y_df,
	split=True,
	test_size=0.5,
	seed=419
)

coeffs.ols(['a', 'b'])  # OLS Calibration using secondary variables a and b
coeffs.theil_sen()  # Theil sen calibration with primary variable only
coeffs.bayesian(['a'], family='Student T') # Bayesian calibration using student t distribution with secondary variable a

coefficients = coeffs.return_coefficients()
test_train = coeffs.return_measurements()

errors = Results(
	test=test_train['Train'],
	train=test_train['Test'],
	coefficients=coeffs['OLS'],
	datasets_to_use=["Calibrated Test", "Uncalibrated Test", "Calibrated Full"],
	x_name="x",
	y_name="y"
	)
errors.max() # Max error
errors.r2() # r2

graphs = Graphs(
	test=test_train['Train'],
	train=test_train['Test'],
	coefficients=coeffs['OLS'],
	datasets_to_use=["Calibrated Test", "Uncalibrated Test", "Calibrated Full"],
	style="ggplot",
	x_name="x",
	y_name="y"
	)
graphs.bland_altman_plot("x vs y (OLS)")  # Bland Altman plot
graphs.save_plots("Output", format="png")  # Save plots as png in Output
```

---

## Acknowledgements

Many thanks to James Murphy at [Mcoding](https://mcoding.io) who's excellent tutorial [Automated Testing in Python](https://www.youtube.com/watch?v=DhUpxWjOhME) and [associated repository](https://github.com/mCodingLLC/SlapThatLikeButton-TestingStarterProject) helped a lot when structuring this package
