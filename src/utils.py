import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from itertools import product


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(train_data, test_data, models, param):
    try:
        report = {}
        i=1

        for model_name,model_class in models.items():
            if model_name == "Exponential Smoothing":

                param_grid = param[model_name]
                keys, values = zip(*param_grid.items())
            
                for v in product(*values):
                    param_dict = dict(zip(keys, v))
                
                    try:
                        model = model_class(train_data, **param_dict)
                        model_fit = model.fit()
                        pred = model_fit.forecast(steps=len(test_data))


                        #predicted_data = pd.DataFrame({'Predicted_Prices': pred}, index=test_data.index)


                    #mae=mean_absolute_error(test_data,pred)
                    #mse=mean_squared_error(test_data,pred)
                    #rmse=np.sqrt(mse)
                        r2=r2_score(test_data,pred)

                        mape = mean_absolute_percentage_error(test_data,pred)
                        #save all models reports
                        report[f"{model_name}_{i}"] = {"class": model_class, "mape": mape, "params": param_dict, "r2": r2}
                        i += 1
                    except Exception as e:
                        raise CustomException(e, sys)
            elif model_name == "SARIMAX":
                
                try:
                    model = model_class(train_data, order=param[model_name]['order'], seasonal_order=param[model_name]['seasonal_order'])
                    model_fit = model.fit(disp=False)
                    pred = model_fit.forecast(steps=len(test_data))

                    #predicted_data = pd.DataFrame({'Predicted_Prices': pred}, index=test_data.index)



                    mape = mean_absolute_percentage_error(test_data,pred)
                    r2=r2_score(test_data,pred)
                    #save all models reports
                    report[f"{model_name}_{i}"] = {"class": model_class, "mape": mape, "params": param [model_name], "r2": r2}
                    i += 1

                except Exception as e:
                    raise CustomException(e, sys)

        return report  

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)