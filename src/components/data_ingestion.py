
import sys
import os
import pandas as pd
from src.exception import CustomException  # Import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation  # Import DataTransformation



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("Enter the Data Ingestion method or component")
        
        try:
            df = pd.read_csv('notebook/Data/StudentsPerformance.csv')
            logging.info("Dataset read as pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)  # Use CustomException here


if __name__ == "__main__":
    logging.info("Starting Data Ingestion")
    try:
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()

        data_transformation = DataTransformation()  # Instantiate DataTransformation
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data Ingestion and Transformation completed")
    except CustomException as e:  # Catch CustomException
        logging.error(f"Custom Exception: {e}")
        print(e)  # Print the error message
    except Exception as e:
        logging.error(f"Exception in main: {e}")
        print(f"An unexpected error occurred: {e}")