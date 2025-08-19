import os
import sys
from dataclasses import dataclass

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array,seasonal_periods=365):
        try:
            logging.info("Initiating model training")

            
            models = {
                "Exponential Smoothing": ExponentialSmoothing,
                "SARIMAX": SARIMAX
             }
            params={
                "Exponential Smoothing": {
                    'trend':['add', 'mul'],
                    'damped_trend':[True, False],
                    'seasonal':['add', 'mul'],
                    'seasonal_periods':[seasonal_periods],
                    'initialization_method':["legacy-heuristic", 'estimated','heuristic',None],
                    'use_boxcox':[False, True]
                },
                "SARIMAX": {
                    'order': (2, 0, 2),
                    'seasonal_order': (0, 1, 1, 12)

                }
                }
                
            logging.info("Models and parameters defined")


            model_report:dict=evaluate_models(train_array,test_array,
                                             models=models,param=params)
            
            logging.info("Models evaluated and report generated")

            ## To get best model score from dict

            best_model_score=min(model_report.items(), key=lambda x: x[1]['mape'])[1]['mape']
            best_model_class= min(model_report.items(), key=lambda x: x[1]['mape'])[1]['class']
            #save the best model and parameters  as an object
            best_model_params = min(model_report.items(), key=lambda x: x[1]['mape'])[1]['params']
            best_model_name = min(model_report.items(), key=lambda x: x[1]['mape'])[0]

            logging.info(f"Best model found model on both training and testing dataset is: {best_model_name} with MAPE score: {best_model_score} and parameters: {best_model_params}")

            best_model=best_model_class(train_array, **best_model_params)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            logging.info("Best model saved")
            #predicting using the best model
            fitted_model=best_model.fit()
            predicted = fitted_model.forecast(steps=len(test_array))



            r2_square = r2_score(test_array, predicted)
            logging.info(f"R2 score of the model is: {r2_square}")
            return r2_square
                   
        except Exception as e:
            raise CustomException(e,sys)