import sys
from dataclasses import dataclass
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DateResampler(BaseEstimator, TransformerMixin):
    def __init__(self, date_col="Dates", freq='D', method='linear'):
        """
        Args:
            date_col (str): name of date column
            freq (str): resampling frequency, e.g. 'D', 'M', 'W'
            method (str): interpolation method, e.g. 'linear', 'spline', 'nearest'
        """
                
        self.date_col = date_col
        self.freq = freq
        self.method = method
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            df = X.copy()
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df = df.set_index(self.date_col)
            df = df.resample(self.freq).interpolate(self.method)
            return df
        except Exception as e:
            raise CustomException(e, sys)

class DataTransformation:
    def __init__(self,freq, method):
        self.data_transformation_config = DataTransformationConfig()
        self.freq = freq
        self.method = method

    def get_data_transformer_object(self):
        try:
            # Create a pipeline with just the date resampler
            preprocessor = Pipeline([
                ("date_resampler", DateResampler(date_col="Dates", freq=self.freq, method=self.method))
            ])
            logging.info("Preprocessor (date resampler) object created")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data is complete")

            preprocessing_obj = self.get_data_transformer_object()

            # Apply preprocessing
            train_processed = preprocessing_obj.fit_transform(train_df)
            test_processed = preprocessing_obj.transform(test_df)

            logging.info("Preprocessing completed")



            train_arr=train_processed[:'2023-12-31']['Prices']
            test_arr=test_processed['2024-01-01':]['Prices']


            # Save the preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessing object saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)

    def general_data_transformation(self, data_path):
            try:
                data = pd.read_csv(os.path.join(data_path, "data.csv"))
                logging.info("Reading data is complete")
    
                preprocessing_obj = self.get_data_transformer_object()
    
                # Apply preprocessing
                data_processed = preprocessing_obj.fit_transform(data)
    
                logging.info("Preprocessing object saved successfully")
    
                return (
                    data_processed
                )
    
            except Exception as e:
                raise CustomException(e, sys)