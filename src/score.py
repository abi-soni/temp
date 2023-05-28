import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def score_model(data, model_folder):
    """
    It loads the model from the model folder, makes predictions on the test data, and
    calculates the RMSE

    :param data: The dataframe to be scored
    :param model_folder: The folder where the model is saved
    :return: The RMSE of the model
    """
    test_data = data.drop("median_house_value", axis=1)
    test_labels = data["median_house_value"].copy()
    model = joblib.load(model_folder + "regressor.pkl")
    final_predictions = model.predict(test_data)
    final_mse = mean_squared_error(test_labels, final_predictions)
    final_rmse = np.sqrt(final_mse)
    logging.debug(f"model rmse score {final_rmse}")
    print(f"RMSE: {final_rmse}")
    return final_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modelpath", help="path for the folder which has train and test files"
    )
    parser.add_argument(
        "datasetpath", help="path for the folder to save the output files"
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
    model_path = parser.parse_args().modelpath
    dataset_path = parser.parse_args().datasetpath
    test_data = pd.read_csv(dataset_path + "test.csv")
    score_model(test_data, model_path)
