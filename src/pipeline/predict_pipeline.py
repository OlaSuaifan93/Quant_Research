import datetime
import sys
import pandas as pd
from src.components import data_transformation
from src.exception import CustomException
from src.utils import load_object
import os
from src.logger import logging
from src.components.data_transformation import DataTransformation


class PredictPipeline:
    def __init__(self, model_path="artifacts/model.pkl"):
        self.model_path = model_path
        self.model = load_object(file_path=self.model_path)

    def predict(self,days=365):
        try:
            model = self.model
            model_fit = model.fit()
            preds = model_fit.forecast(steps=days)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(  self,
        Date: datetime.datetime,
        Prices: float):

        self.Date = Date
        self.Prices = Prices

   

    def get_future_data_as_data_frame(self,Days=5,start_date="2024-10-01",freq='D'):
        try:
            start_date_dt = pd.to_datetime(start_date)
            end_date=start_date_dt + pd.Timedelta(days=Days-1)
            future_dates= pd.date_range(start=start_date_dt, end=end_date, freq=freq)

            predictor= PredictPipeline()
            future_prices = predictor.predict(days=Days)

            custom_data_input_dict = {
                "Dates": future_dates,
                "Prices": future_prices
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
    def estimate_gas_price(self,date):
        try:
            data_transformation = DataTransformation(freq='D', method='linear')

            past_data = data_transformation.general_data_transformation('artifacts')
            past_data.reset_index(inplace=True, drop=False)
            past_data['Dates'] = pd.to_datetime(past_data['Dates'])

            date_dt = pd.to_datetime(date)
            last_date = past_data['Dates'].max()

            days=(date_dt-last_date).days
            start_date= last_date + pd.Timedelta(days=1)
            future_data=self.get_future_data_as_data_frame(Days=days,start_date=start_date)

            combined_data = pd.concat([past_data, future_data], ignore_index=False)
            combined_data.reset_index(drop=True, inplace=True)

            data=combined_data[combined_data['Dates']==date_dt]
            
            return data["Prices"].values[0]
            
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example usage
    from datetime import datetime

    custom_data = CustomData(Date=datetime.now(), Prices=0)

    # Generate future data
    future_df = custom_data.get_future_data_as_data_frame(Days=365, start_date="2024-10-01")
    

    # Estimate gas price for a specific date
    price = custom_data.estimate_gas_price("2024-10-03")
    print(price)
