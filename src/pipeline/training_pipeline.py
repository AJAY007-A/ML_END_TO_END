import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher


class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("======= STARTING TRAINING PIPELINE =======")

            # Stage 1: Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # Stage 2: Data Validation
            validator = DataValidation()
            if not validator.validate_dataset(train_path):
                raise CustomException("Data validation failed for training data.", sys)

            # Stage 3: Data Transformation
            transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
                train_path=train_path, test_path=test_path
            )

            # Stage 4: Model Training
            trainer = ModelTrainer()
            best_model_name, best_score = trainer.initiate_model_trainer(
                train_array=train_arr, test_array=test_arr
            )

            # Stage 5: Model Evaluation
            model_path = trainer.model_trainer_config.trained_model_file_path
            evaluator = ModelEvaluation()
            metrics = evaluator.initiate_model_evaluation(
                test_array=test_arr, model_path=model_path
            )

            # Stage 6: Model Pusher
            pusher = ModelPusher()
            pusher.initiate_model_pusher(
                model_path=model_path, preprocessor_path=preprocessor_path
            )

            logging.info("======= TRAINING PIPELINE COMPLETED SUCCESSFULLY =======")
            print(f"Training Pipeline Complete!")
            print(f"Best Model: {best_model_name}")
            print(f"Test R2 Score: {metrics['R2_Score']:.4f}")
            print(f"Test MAE: {metrics['MAE']:.2f}")
            print(f"Test RMSE: {metrics['RMSE']:.2f}")

            return metrics
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
