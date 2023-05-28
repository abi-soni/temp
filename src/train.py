import argparse
import logging

import joblib
import pandas as pd

from pipeline import build_pipeline


def train_model(data, output_folder):
    """
    It takes the data and the output folder as input, drops the median_house_value
    column from the data, and then uses the remaining columns to train a model. It
    then saves the model in the output folder

    :param data: The dataframe containing the data to train the model on
    :param output_folder: The folder where the model will be saved
    """
    train_data = data.drop("median_house_value", axis=1)
    train_labels = data["median_house_value"].copy()
    pipeline = build_pipeline()
    pipeline.fit(train_data, train_labels)
    joblib.dump(pipeline, output_folder + "regressor.pkl")
    logging.debug(
        f"model trained and saved at { output_folder}regressor.pkl successfully"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputpath", help="path for the folder which has train and test files"
    )
    parser.add_argument(
        "outputpath", help="path for the folder to save the output files"
    )
    parser.add_argument(
        "--noconsolelog", help="path for the log folder", action="store_true"
    )
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
    input_path = parser.parse_args().inputpath
    output_path = parser.parse_args().outputpath
    train_data = pd.read_csv(input_path + "train.csv")
    train_model(train_data, output_path)
