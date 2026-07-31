import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging


class DataValidation:
    def __init__(self):
        self.required_columns = [
            "car_name",
            "vehicle_age",
            "km_driven",
            "seller_type",
            "fuel_type",
            "transmission_type",
            "mileage",
            "engine",
            "max_power",
            "seats",
            "selling_price",
        ]

    def validate_dataset(self, file_path: str) -> bool:
        try:
            logging.info(f"Initiating Data Validation for file: {file_path}")
            df = pd.read_csv(file_path)

            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                logging.error(f"Data Validation failed: Missing columns {missing_cols}")
                return False

            if df.empty:
                logging.error("Data Validation failed: Dataset is empty.")
                return False

            logging.info("Data Validation passed successfully.")
            return True
        except Exception as e:
            raise CustomException(e, sys)
