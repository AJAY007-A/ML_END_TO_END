import sys
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from src.exception import CustomException
from src.logger import logging
from src.config import ModelTrainerConfig
from src.utils import save_object, evaluate_models


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data for model training...")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Random Forest": RandomForestRegressor(random_state=42),
            }

            logging.info("Evaluating candidate regressor models...")
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            # Find best model based on test R2 score
            best_model_name = max(
                model_report, key=lambda k: model_report[k]["test_r2"]
            )
            best_model_info = model_report[best_model_name]
            best_model = best_model_info["model"]
            best_model_score = best_model_info["test_r2"]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R2 score >= 0.6", sys)

            logging.info(
                f"Best model selected: [{best_model_name}] with R2 score: {best_model_score:.4f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return best_model_name, best_model_score
        except Exception as e:
            raise CustomException(e, sys)
