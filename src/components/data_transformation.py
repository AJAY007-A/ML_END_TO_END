import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.config import DataTransformationConfig
from src.utils import save_object


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_cols, categorical_cols):
        try:
            logging.info(f"Numerical columns for transformation: {numerical_cols}")
            logging.info(f"Categorical columns for transformation: {categorical_cols}")

            num_pipeline = StandardScaler()
            cat_pipeline = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully for transformation.")

            drop_columns = ["Unnamed: 0", "brand", "model"]
            for col in drop_columns:
                if col in train_df.columns:
                    train_df.drop(columns=[col], inplace=True)
                if col in test_df.columns:
                    test_df.drop(columns=[col], inplace=True)

            target_column_name = "selling_price"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            numerical_cols = input_feature_train_df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            categorical_cols = input_feature_train_df.select_dtypes(
                include=["object"]
            ).columns.tolist()

            preprocessing_obj = self.get_data_transformer_object(
                numerical_cols=numerical_cols, categorical_cols=categorical_cols
            )

            logging.info("Fitting and transforming train and test input features...")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object...")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
