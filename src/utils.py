import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Saved object successfully at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict) -> dict:
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            report[name] = {
                "train_r2": train_r2,
                "test_r2": test_r2,
                "mae": mae,
                "rmse": rmse,
                "model": model,
            }
            logging.info(f"Model [{name}] - Test R2: {test_r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        return report
    except Exception as e:
        raise CustomException(e, sys)
