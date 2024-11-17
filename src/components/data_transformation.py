import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        logging.info('Data Transformation Stage Started')
        try:
            # Define numerical and categorical columns
            numerical_cols = ['reading_score', 'writing_score']
            categorical_cols = [
                'gender', 'race_ethnicity', 'parental_level_of_education', 
                'lunch', 'test_preparation_course'
            ]
            
            # Define pipelines for numerical and categorical columns
            num_Pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
                    ('scaler', StandardScaler())  # Scale the data
                ]
            )
            cat_Pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical values
                ]
            )
            
            # Combine pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num_Pipeline', num_Pipeline, numerical_cols),
                ('cat_Pipeline', cat_Pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_Transformation(self, train_path, test_path):
        try:
            # Load the training and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading data as pandas DataFrame')

            logging.info(f'Train data head:\n{train_df.head()}')
            logging.info(f'Test data head:\n{test_df.head()}')

            # Get the preprocessor object
            preprocessor_obj = self.get_data_transformation_obj()
            logging.info('Preprocessor object created')

            # Define the target column
            target_column_name = 'math_score'

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying transformations on training and test data")

            # Transform input features using the preprocessor
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine transformed features and target values
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object for later use
            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("Preprocessor object saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occurred during data transformation")
            raise CustomException(e, sys)
