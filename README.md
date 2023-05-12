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

calidhayte calibrates one set of measurements against another, using a variety of parametric and non parametric techniques.
The datasets are split by k-fold cross validation and stratified so the distribution of 'true' measurements is consistent in all.
It can then performs multiple error calculations to validate them, as well as produce several graphs to visualise the calibrations. 

---

## Main Features

- Calibrate one set of measurements (cross-comparing all available secondary variables) against a 'true' set
	- A suite of calibration methods are available, including bayesian regression
- Perform a suite of error calculations on the resulting calibration
- Visualise results of calibration
- Summarise calibrations to highlight best performing techniques

---

## How to install

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

Please see [requirements.txt](./requirements.txt).

---

## Example Usage

This module requires two dataframes as a prerequisite. 

**Independent Measurements**
||x|a|b|c|d|e|
|---|---|---|---|---|---|---|
|2022-01-01|0.1|0|7|2.2|3|5|
|2022-01-02|0.7|1|3|2|8.9|1|
|2022-01-03|*nan*|*nan*|1|*nan*|*nan*|7|
|_|_|_|_|_|_|_|
|2022-09-30|0.5|3|1|2.7|4|0|

**Dependent Measurements**
||x|
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
from calidhayte import Calibrate, Results, Graphs, Summary

# x_df is a dataframe containing multiple columns containing independent measurements.
# The primary measurement is denoted by the 'Values' columns, the other measurement columns can have any name.
# y_df is a dataframe containing the dependent measurement in the 'Values' column.

coeffs = Calibrate(
	x=x_df,
	y=y_df
	target='x'
)

cal.linreg()
cal.theil_sen()
cal.random_forest(n_estimators=500, max_features=1.0)

models = coeffs.return_models()

results = Results(
	x=x_df,
	y=y_df,
	target='x',
	models=models
)

results.r2()
results.median_absolute()
results.max()

results_df = results.return_errors()
results_df.to_csv('results.csv')

graphs = Graphs(
	x=x_df,
	y=y_df,
	target='x',
	models=models,
	x_name='x',
	y_name='y'
)
graphs.ecdf_plot()
graphs.lin_reg_plot()
graphs.save_plots()

```

---

## Acknowledgements

Many thanks to James Murphy at [Mcoding](https://mcoding.io) who's excellent tutorial [Automated Testing in Python](https://www.youtube.com/watch?v=DhUpxWjOhME) and [associated repository](https://github.com/mCodingLLC/SlapThatLikeButton-TestingStarterProject) helped a lot when structuring this package
