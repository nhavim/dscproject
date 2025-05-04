import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessing_object_file_path = os.path.join('artifacts', 'preprocessor.pkl')
class DataTransformation:
    """
    Data Transformation Configuration
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This method returns the configuration for data transformation.
        '''
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender", 
                "race/ethnicity", 
                "parental level of education",
                "lunch", 
                "test preparation course"
            ]

            # Define the numerical and categorical pipelines
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder()),
                ('scaler', StandardScaler())
                ]
            ) 
            
            # Logging the creation of pipelines
            logging.info(f"Numerical columns: {categorical_columns}")
            logging.info(f"Categorical columns: {numerical_columns}")
            
            # Combine the numerical and categorical pipelines into a preprocessor
            # using ColumnTransformer
            # ColumnTransformer to apply different transformations to different columns
            # Preprocessor for numerical and categorical features
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)   
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This method initiates the data transformation process.
        """
        try:
            # Read the train and test data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("The train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column="math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df=train_data.drop(columns=[target_column], axis=1)
            target_feature_train_df=train_data[target_column]

            input_feature_test_df=test_data.drop(columns=[target_column], axis=1)
            target_feature_test_df=test_data[target_column] 

            logging.info(
                "Applying preprocessing object on traing data and test data"
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)

            ]

            test_arr= np.c_[
                 input_feature_test_df, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessing_object_file_path,
                obj=preprocessing_obj 
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_object_file_path,
            )
        except Exception as e:
            raise CustomException(sys,e)  