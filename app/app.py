#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os
import tarfile
import urllib.request
import warnings
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):

    """Downloading the raw dataset.


    Parameters
    ----------
    housing_url  : str
        Path to download housing url.
    housing_path: str
        Path to store the new dataset.


    Returns
    --------
    bool
        True if successful, False otherwise.
    """

    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    return True


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Load the raw dataset.


    Parameters
    ----------
    csv_path  : str
        Path to fetch the housing dataset
    Returns
    --------
    DataFrame Object
        Returns a pandas object with housing dataset
    """

    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Deriving new attributes and adding them to the dataframe.


    Parameters
    --------
    BaseEstimator: obj
        Function from sklearn to implement set_params and get_params.
    TransformerMixin: obj
        Function from skelarn to implement fit_transform method.


    """

    def __init__(self, indexes_, add_bedrooms_per_room=True):

        """
        Saving the indexes of the attributes and checking if the bedrooms_per_room attribute should be added.


        Parameters
        --------
        indexes_: tuple
            A tuple of indexes of the columns.
        add_bedrooms_per_room: bool
            True if the bedrooms_per_room column should be added, False otherwise.


        Returns:
        --------
        None
        """
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.indexes_ = indexes_

    def fit(self, X, y=None):
        """
        Fitting the dataframe.


        Parameters
        --------
        X: DataFrame Object
            DataFrame to be transformed.


        Returns:
        --------
        None
        """

        return self  # nothing else to do

    def transform(self, X, y=None):
        """
        Deriving and adding the new columns to the dataframe.


        Parameters
        --------
        X: DataFrame Object
            DataFrame to be transformed.


        Returns:
        --------
            Transformed dataframe with added columns.
        """

        (rooms_ix, bedrooms_ix, population_ix, households_ix) = self.indexes_
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]

        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def arguments():
    """
    Function to parse the arguments.


    Parameters
    ----------
    None
    Returns
    -------
    Args
        Returns the arguments added in the argument parser
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--log-level', 
        default='INFO', 
        help='Log level (default: %(default)s)'
    )
    
    parser.add_argument(
        '--log-path',
        help='Path to the log file'
    )
    
    parser.add_argument(
        '--no-console-log',
        action='store_true',
        help='Disable console logging'
    )
    
    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="maximum features parameter for random forest",
    )
    
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="n estimators parameter for random forest",
    )

    parser.add_argument(
        "--max_features_range",
        action="extend",
        type=int,
        nargs="+",
        default=[],
        help="maximum features range for gridsearch cv",
    )

    parser.add_argument(
        "--n_estimators_range",
        action="extend",
        default=[],
        type=int,
        nargs="+",
        help="n_estimators range for gridsearch cv",
    )

    parser.add_argument(
        "--imputing_strategy",
        type=str,
        default="median",
        help="imputing startegy for misiing values",
    )

    return parser.parse_args()


def stratified_split(housing):
    """
    Stratified sampling and splitting the dataset into test and train set.


    Parameters
    ----------
    housing: DataFrame Object

        Dataframe to be split into test and train


    Returns
    -------
    strat_train_set: DataFrame Object

        Returns the train dataset.

    strat_test_set: DataFrame Object

        Returns the test dataset.
    """

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[
            0.0,
            1.5,
            3.0,
            4.5,
            6.0,
            np.inf,
        ],
        labels=[1, 2, 3, 4, 5],
    )

    # splitting the dataset into train and test based on income cateory column

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for (train_index, test_index) in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

        # dropping the income cat column

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return (strat_train_set, strat_test_set)


def train(
    strat_train_set,
    imputing_strategy,
    max_features,
    n_estimators,
    max_features_range,
    n_estimators_range,
):
    """
    Data Preparation and training the model.


    Parameters
    ----------
    strat_train_set: DataFrame Object
        Dataset to train the model.

    imputing_strategy: string
        Strategy to impute missing values.

    max_features: int
        Maximum features to be used by the random forest model.

    n_estimators: int
        Number of estimators to be used by the random forest model.

    max_features_range: list
        List of max_features values to be passed in grid search cv.

    n_estimators_range: list
        list of n_estimators value to be passed in grid search cv.


    Returns
    -------
    max_features: int
        Returns max_features value (Same if passed as an input else obtained from grid search cv).

    n_estimators: int
        Returns n_estimators value (Same if passed as an input else obtained from grid search cv).

    max_features_range: list
        Returns the list of max_features value used in grid search cv. None if grid search not run.

    n_estimators_range: list
        Returns the list of n_estimators values used in grid search cv. None if grid search not run.

    rf_reg_pipeline: model object

        Returns the trained random forest model.
    ,
    """

    # defining the depenedent and independent variable

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # storing the indexes of the below columns in the data frame
    # rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
    # attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    # housing_extra_attribs = attr_adder.transform(housing.values)

    np.random.seed(40)
    col_names = ("total_rooms", "total_bedrooms", "population", "households")

    # storing the indexes of the columns

    indexes_ = tuple([housing.columns.get_loc(c) for c in col_names])

    # pipeline for numerical column transformations

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("attribs_adder", CombinedAttributesAdder(indexes_)),
            ("std_scaler", StandardScaler()),
        ]
    )

    # defining categorical and numerical columns for applying the
    # transformations seperately

    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num.columns)
    cat_attribs = ["ocean_proximity"]

    # full pipeline including the transformations on both numerical and categorucal columns

    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs)]
    )

    # setting the imputing strategy parameter in the full pipeline

    full_pipeline.set_params(num__imputer__strategy=imputing_strategy)

    # intializing the random forest

    random_forest = RandomForestRegressor()

    # creating a pipeline that includes both the full pipeline and the model

    rf_reg_pipeline = Pipeline(
        [("Data_Prep", full_pipeline), ("RandomForest", random_forest)]
    )

    # if any of max_features and n_estimators is not specified then run the gridsearch cv

    if max_features is None or n_estimators is None:

        if max_features is not None:

            # if max_features is specified then passing that value as the max_features_range

            max_features_range = max_features

        if n_estimators is not None:

            # if n_estimators is specified then passing that value as the n_estimators_range

            n_estimators_range = n_estimators

        if max_features_range == []:
            max_features_range = [4, 6, 8, 10, 12]

        if n_estimators_range == []:
            n_estimators_range = [80, 100, 150, 200]

        param_grid = [
            {
                "RandomForest__n_estimators": n_estimators_range,
                "Data_Prep__cat__handle_unknown": ["infrequent_if_exist"],
                "RandomForest__max_features": max_features_range,
            }
        ]

        grid_search = GridSearchCV(
            rf_reg_pipeline,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )

        # training the grid_search model
        grid_search.fit(housing, housing_labels)

        # storing the best parameters

        max_features = grid_search.best_params_["RandomForest__max_features"]
        n_estimators = grid_search.best_params_["RandomForest__n_estimators"]
    else:

        # if both the random forest parameters are passed as a command line argument the grid search won't be run

        max_features_range = None
        n_estimators_range = None

    np.random.seed(40)

    # setting the random forest parameters

    rf_reg_pipeline.set_params(
        RandomForest__max_features=max_features, RandomForest__n_estimators=n_estimators
    )

    # training the model

    rf_reg_pipeline.fit(housing, housing_labels)
    
    with open(os.path.join("../../app", "final_model.pickle"), "wb") as f:
        pickle.dump(rf_reg_pipeline, f)

    return (
        max_features,
        n_estimators,
        max_features_range,
        n_estimators_range,
        rf_reg_pipeline,
    )


def scoring(test_set, model):
    """
    Evaluating the r2 and rmse score.


    Parameters
    ----------
    test_set: DataFrame Object
        Dataset to evaluate the scores on.

    model: Model Object
        Trained Random Forest Model.


    Returns
    -------
    r2: float

        Returns the r2_score value.

    final_rmse: float

        Returns the rmse value.
    """

    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()

    final_predictions = model.predict(X_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    r2 = r2_score(y_test, final_predictions)
    print(f"r2 score is: {r2}")
    print(f"rmse value is: {final_rmse}")

    return (r2, final_rmse)


def main(exp_name):
    """
    Storing the parameters,metrics and model using mlflow after running the train function.


    Parameters
    ----------
    exp_name: string
        Experiment name under which the paramaters, metrics and model to be logged.
        Can be an experiment to be created or an already existing experiment.


    Returns
    -------
    None

    """

    # storing the args passed in the command line

    args = arguments()
    
    # Configure logging
    if args.log_level == 'DEBUG':
        # Set the log level to debug
        logging.basicConfig(level=logging.DEBUG)

    if args.log_path:
        # Set the log file path
        logging.basicConfig(filename=args.log_path)

    if not args.no_console_log:
        # Enable console logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(args.log_level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logging.getLogger().addHandler(console_handler)
        
   
    
    # Log a message
    logging.info('Starting script')
    imputing_strategy = args.imputing_strategy
    n_estimators = args.n_estimators
    max_features = args.max_features
    n_estimators_range = args.n_estimators_range
    max_features_range = args.max_features_range

    # creates a new experiment if it does not exist

    details = mlflow.set_experiment(exp_name)
    experiment_id = details.experiment_id

    # starting a parent run

    with mlflow.start_run(
        run_name="PARENT_RUN", experiment_id=experiment_id, description="parent"
    ) as parent_run:

        # downloading the dataset

        try:
            fetch_housing_data()
            
        except Exception as e:

            print(f"Error occurred: {e}")

        housing = load_housing_data()
        # mlflow.log_param("parent", "yes")

        # starting the nested child run

        with mlflow.start_run(
            run_name="training",
            experiment_id=experiment_id,
            description="training",
            nested=True,
        ) as child_run:

            # splitting the dataset

            (train_set, test_set) = stratified_split(housing)

            # training the model

            (
                max_features,
                n_estimators,
                max_features_range,
                n_estimators_range,
                model,
            ) = train(
                train_set,
                imputing_strategy,
                max_features,
                n_estimators,
                max_features_range,
                n_estimators_range,
            )

            # loggin the parameters

            mlflow.log_param("max_features", max_features)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("n_estimators_range", n_estimators_range)
            mlflow.log_param("max_features_range", max_features_range)
            mlflow.log_param("imputing_strategy", imputing_strategy)

        # starting the nested child run for storing the scores

        with mlflow.start_run(
            run_name="scoring",
            experiment_id=experiment_id,
            description="scoring",
            nested=True,
        ) as child_run:

            # evaluating the scores

            (r2, rmse) = scoring(test_set, model)

            # logging the scores and the model

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    
    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(model, "model", registered_model_name="HousePricePredictionModel")
    else:
        mlflow.sklearn.log_model(model, "model")        # mlflow.sklearn.log_model(model, "model")
            
    

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    main("experiment1")

    
