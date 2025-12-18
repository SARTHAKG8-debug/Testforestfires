from flask import Flask, request, jsonify,render_template
import pickle
import numpy as np  
import pandas as pd
from sklearn.preprocessing import StandardScaler    

application = Flask(__name__)
app = application

#import ridge regressor and scaler
ridge_model = pickle.load(open("ridge.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method=='POST':
        #fetching the data from the form
        DMC = float(request.form['DMC'])
        Classes = float(request.form['Classes'])
        ISI = float(request.form['ISI'])
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        WS = float(request.form['WS'])
        Rain = float(request.form['Rain'])
        Regions = int(request.form['Regions'])
        FFMC = float(request.form['FFMC'])
        
        #scaling the data
        new_data_scaled =scaler.transform([[ DMC, Classes, ISI, Temperature, RH, WS, Rain, Regions, FFMC]])
        
        #predicting the FWI value using ridge regression model
        fwi_prediction = ridge_model.predict(new_data_scaled)
        return render_template('home.html', result=fwi_prediction[0])
        
    else:
        return render_template('home.html')
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0") 

