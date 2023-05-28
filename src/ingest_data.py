import argparse
import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data", "raw")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    > It downloads the housing.tgz file from the housing_url, extracts the housing.csv
    file from the housing.tgz file, and saves the housing.csv file in the housing_path
    directory

    :param housing_url: The URL of the housing dataset (defaults to the one hosted by
    the University of California, Irvine)
    :param housing_path: The directory to save the dataset in
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    logging.debug(f"Source file extracted at {tgz_path} successfully")
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    It loads the housing data from the given path, and returns a Pandas DataFrame object
    containing the data

    :param housing_path: The path to the housing dataset
    :return: A dataframe
    """
    """
    It loads the housing data from the given path, and returns a Pandas DataFrame object
    containing the data
    
    :param housing_path: The path to the housing dataset
    :return: A dataframe
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def check_nulls(data):
    """
    If there are any null values in the data, return False. Otherwise, return True

    :param data: The dataframe you want to check for null values
    :return: A boolean value.
    """
    if data.isnull().values.any():
        return False
    else:
        return True


def prepare_train_test_data(housing):
    """
    - Check for null values in the dataframe and drop them if any.
    - Create a new column called `income_cat` which is a categorical variable based on
    the `median_income` column.
    - Split the data into train and test sets using the `StratifiedShuffleSplit` class.
    - Drop the `income_cat` column from the train and test sets

    :param housing: The dataframe that we want to split into train and test sets
    :return: A dictionary with two keys, train and test, each of which contains a data
    frame.
    """
    if check_nulls(housing):
        housing.dropna(axis=0, how="any", inplace=True)
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    logging.debug("Train test data computed successfully")

    return {"train": strat_train_set, "test": strat_test_set}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path for the folder to save the processed files")
    parser.add_argument(
        "--noconsolelog", help="path for the log folder", action="store_true"
    )
    processed_data_folder = parser.parse_args().path
    if parser.parse_args().noconsolelog:
        logging.basicConfig(
            filename="logs/logs.log",
            level=logging.DEBUG,
            format="%(pathname)s:%(levelname)s:%(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(pathname)s:%(levelname)s:%(message)s",
        )
    fetch_housing_data()
    raw_data = load_housing_data()
    complete_data = prepare_train_test_data(raw_data)
    complete_data["train"].to_csv(processed_data_folder + "train.csv")
    complete_data["test"].to_csv(processed_data_folder + "test.csv")
    logging.debug(f"Train test data saved at {processed_data_folder} folder succesfully")
