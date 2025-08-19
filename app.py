from datetime import datetime
from sqlite3 import Date
from flask import Flask,request,render_template
import numpy as np
import pandas as pd  

from sklearn.preprocessing import StandardScaler 
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

from flask import Flask, request, render_template

from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Get the date input from the form
        date_input = request.form.get('Date')

        # Create a CustomData object (Prices can be dummy)
        custom_data = CustomData(Date=datetime.now(), Prices=0)
        #data=custom_data.get_future_data_as_data_frame(Days=365,start_date=date_input)

        try:
            # Estimate gas price for the selected date
            result = custom_data.estimate_gas_price(date_input)
            return render_template('home.html', results=result)
        except Exception as e:
            # If prediction fails, show error message
            return render_template('home.html', results="Error fetching prediction.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


 