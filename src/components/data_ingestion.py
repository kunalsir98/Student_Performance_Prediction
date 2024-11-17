import os 
import sys
import pandas as pd 
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Stage Started')

        try:
            df=pd.read_csv(os.path.join('notebook/data.csv'))
            logging.info('DataFrame read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Spliting of DataStarted')

            train_set,test_set=train_test_split(df,test_size=0.32)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data ingestion stage completed')

            

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        
        except Exception as e:
            logging.info('Exception Occurred at data Ingestion Stage')
            raise CustomException(e,sys)
