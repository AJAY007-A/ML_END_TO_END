import sys
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.exception import CustomException
from src.logger import logging
from src.config import ModelEvaluationConfig
from src.utils import load_object


class ModelEvaluation:
    def __init__(self):
        self.eval_config = ModelEvaluationConfig()

    def initiate_model_evaluation(self, test_array, model_path: str):
        try:
            logging.info("Initiating Model Evaluation step...")
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model = load_object(model_path)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            metrics = {
                "MAE": float(mae),
                "MSE": float(mse),
                "RMSE": float(rmse),
                "R2_Score": float(r2),
            }

            logging.info(f"Evaluation Metrics: {metrics}")

            with open(self.eval_config.metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logging.info(f"Saved evaluation metrics to {self.eval_config.metrics_file_path}")
            return metrics
        except Exception as e:
            raise CustomException(e, sys)
