import os
import sys
from src.exception import CustomException
from src.logger import logging


class ModelPusher:
    def __init__(self):
        pass

    def initiate_model_pusher(self, model_path: str, preprocessor_path: str) -> bool:
        try:
            logging.info("Initiating Model Pusher readiness check...")
            if not os.path.exists(model_path):
                raise CustomException(f"Trained model file not found at {model_path}", sys)

            if not os.path.exists(preprocessor_path):
                raise CustomException(f"Preprocessor object file not found at {preprocessor_path}", sys)

            logging.info(f"Model pusher verified artifacts at {model_path} and {preprocessor_path}. Status: READY.")
            return True
        except Exception as e:
            raise CustomException(e, sys)
