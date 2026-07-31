import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading model and preprocessor objects...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Transforming input features with preprocessor...")
            data_scaled = preprocessor.transform(features)

            logging.info("Making prediction...")
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        car_name: str,
        vehicle_age: int,
        km_driven: int,
        seller_type: str,
        fuel_type: str,
        transmission_type: str,
        mileage: float,
        engine: float,
        max_power: float,
        seats: int,
    ):
        self.car_name = car_name
        self.vehicle_age = vehicle_age
        self.km_driven = km_driven
        self.seller_type = seller_type
        self.fuel_type = fuel_type
        self.transmission_type = transmission_type
        self.mileage = mileage
        self.engine = engine
        self.max_power = max_power
        self.seats = seats

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "car_name": [self.car_name],
                "vehicle_age": [int(self.vehicle_age)],
                "km_driven": [int(self.km_driven)],
                "seller_type": [self.seller_type],
                "fuel_type": [self.fuel_type],
                "transmission_type": [self.transmission_type],
                "mileage": [float(self.mileage)],
                "engine": [float(self.engine)],
                "max_power": [float(self.max_power)],
                "seats": [int(self.seats)],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"Custom data converted to DataFrame:\n{df}")
            return df
        except Exception as e:
            raise CustomException(e, sys)
