import re
from typing import Literal

import pandas as pd


def prepare_datasets(
        datasets_to_use: list[
            Literal[
                "Calibrated Train",
                "Calibrated Test",
                "Calibrated Full",
                "Uncalibrated Train",
                "Uncalibrated Test",
                "Uncalibrated Full",
                "Minimum",
                "Maximum"
                ]
            ],
        train: pd.DataFrame,
        test: pd.DataFrame,
        y_full: dict[str, pd.DataFrame],
        y_subsets: dict[str, pd.DataFrame]
        ) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Prepare the datasets to be analysed

    Parameters
    ----------
    datasets_to_use : list[str]
        List containing datasets to use
        Datasets are as follows:
            - Calibrated Train
                predicted y measurements from training set using
                coefficients
            - Calibrated Test
                predicted y measurements from testing set using
                coefficients
            - Calibrated Full
                predicted y measurements from both sets using
                coefficients
            - Uncalibrated Train
                Base x measurements from training set
            - Uncalibrated Test
                Base x measurements from testing set
            - Uncalibrated Full
                Base x measurements from testing set
            - Minimum
                Mean bayesian coefficients - 2 times standard deviation
            - Maximum
                Mean bayesian coefficients + 2 times standard deviation
    Returns
    -------
    dict[str, dict[str, pd.DataFrame]]
        Contains all subsets of the data to be analysed
    """
    uncalibrated_datasets = {
            "Uncalibrated Train": {
                "x": train.loc[:, ['x']],
                "y": train.loc[:, ['y']]
                },
            "Uncalibrated Test": {
                "x": test.loc[:, ['x']],
                "y": test.loc[:, ['y']]
                },
            "Uncalibrated Full": {
                "x": y_full['Uncalibrated'].loc[:, ['x']],
                "y": y_full['Uncalibrated'].loc[:, ['y']]
                }
            }
    pymc_bool = all(
            [
                any(["Mean." in key for key in y_subsets.keys()]),
                any(["Minimum." in key for key in y_subsets.keys()]),
                any(["Maximum." in key for key in y_subsets.keys()])
                ]
            )
    if pymc_bool:
        cal_datasets = {
                "Calibrated Train (Mean)": {
                    "x": y_subsets['Mean.Train'],
                    "y": train.loc[:, ['y']]
                    },
                "Calibrated Test (Mean)": {
                    "x": y_subsets['Mean.Test'],
                    "y": test.loc[:, ['y']]
                    },
                "Calibrated Full (Mean)": {
                    "x": y_full['Mean.Calibrated'],
                    "y": y_full['Uncalibrated'].loc[:, ['y']]
                    },
                "Calibrated Train (Minimum)": {
                    "x": y_subsets['Minimum.Train'],
                    "y": train.loc[:, ['y']]
                    },
                "Calibrated Test (Minimum)": {
                    "x": y_subsets['Minimum.Test'],
                    "y": test.loc[:, ['y']]
                    },
                "Calibrated Full (Minimum)": {
                    "x": y_full['Minimum.Calibrated'],
                    "y": y_full['Uncalibrated'].loc[:, ['y']]
                    },
                "Calibrated Train (Maximum)": {
                    "x": y_subsets['Maximum.Train'],
                    "y": train.loc[:, ['y']]
                    },
                "Calibrated Test (Maximum)": {
                    "x": y_subsets['Maximum.Test'],
                    "y": test.loc[:, ['y']]
                    },
                "Calibrated Full (Maximum)": {
                    "x": y_full['Maximum.Calibrated'],
                    "y": y_full['Uncalibrated'].loc[:, ['y']]
                    },
                }
    else:
        cal_datasets = {
                "Calibrated Train": {
                    "x": y_subsets['Train'],
                    "y": train.loc[:, ['y']]
                    },
                "Calibrated Test": {
                    "x": y_subsets['Test'],
                    "y": test.loc[:, ['y']]
                    },
                "Calibrated Full": {
                    "x": y_full['Calibrated'],
                    "y": y_full['Uncalibrated'].loc[:, ['y']]
                    }
                }
    datasets = uncalibrated_datasets | cal_datasets
    uncal_sets = filter(
            lambda x: bool(re.search(r'^Uncalibrated ', x)),
            datasets_to_use
            )
    cal_sets = filter(
            lambda x: bool(re.search(r'^Calibrated ', x)),
            datasets_to_use
            )
    selected_datasets = dict()
    for uncal_key in uncal_sets:
        selected_datasets[str(uncal_key)] = datasets[uncal_key]
    if pymc_bool:
        min_max_sets = list(

                filter(
                    lambda x: bool(re.search(r'^Minimum|^Maximum', x)),
                    datasets_to_use
                )
            )
        for cal_key in cal_sets:
            for pymc_subset in ['Mean'] + min_max_sets:
                selected_datasets[
                        f'{cal_key} ({pymc_subset})'
                        ] = datasets[f'{cal_key} ({pymc_subset})']
    else:
        for cal_key in cal_sets:
            selected_datasets[str(cal_key)] = datasets[cal_key]

    return selected_datasets
