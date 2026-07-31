from dataclasses import dataclass
import os


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    source_dataset_path: str = os.path.join("Dataset", "cardekho_dataset.csv")


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


@dataclass
class ModelEvaluationConfig:
    metrics_file_path: str = os.path.join("artifacts", "evaluation_metrics.json")
