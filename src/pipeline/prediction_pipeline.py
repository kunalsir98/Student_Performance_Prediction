import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)

            return pred


        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 reading_score: float,
                 writing_score: float,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str):
        """
        Initialize the input features.

        Args:
            reading_score (float): Score in reading.
            writing_score (float): Score in writing.
            gender (str): Gender of the student.
            race_ethnicity (str): Race/ethnicity group.
            parental_level_of_education (str): Parent's education level.
            lunch (str): Type of lunch (e.g., standard or free/reduced).
            test_preparation_course (str): Whether a test prep course was completed.
        """
        self.reading_score = reading_score
        self.writing_score = writing_score
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course

    def get_data_as_dataframe(self):
        """
        Converts the input data into a pandas DataFrame for prediction.

        Returns:
            pd.DataFrame: DataFrame containing the input features.
        """
        try:
            custom_data_input_dict = {
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score],
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame created successfully for prediction.")
            return df
        except Exception as e:
            logging.error("Exception occurred while creating DataFrame.")
            raise CustomException(e, sys)
