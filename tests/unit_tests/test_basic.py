import pandas as pd

from src.ingest_data import check_nulls
from src.pipeline import build_pipeline


def test_null_values():
    """
    It checks if there are any null values in the dataframe
    """
    data_path = "data/processed/"
    test_data = pd.read_csv(data_path + "test.csv")
    train_data = pd.read_csv(data_path + "train.csv")
    train_flag = check_nulls(train_data)
    test_flag = check_nulls(test_data)
    flag = train_flag and test_flag
    assert not flag


def test_functional():
    """
    It checks if pipeline is built or not
    """
    Pipeline = build_pipeline()
    assert Pipeline
