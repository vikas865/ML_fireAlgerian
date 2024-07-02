import pickle
from flask import Flask, render_template, jsonify,request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)


model=pickle.load(open('models/ridge_lr.pkl','rb'))

scalar=pickle.load(open('models/scalar_ridge.pkl','rb'))
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature= float(request.form.get('Temperature'))
        RH= float(request.form.get('RH'))
        Ws= float(request.form.get('Ws'))
        Rain= float(request.form.get('Rain'))
        FFMC= float(request.form.get('FFMC'))
        DMC= float(request.form.get('DMC'))
        ISI= float(request.form.get('ISI'))
        Classes= float(request.form.get('Classes'))
        Region= float(request.form.get('Region'))

        new_data_scaled=scalar.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes,
       Region]])
        
        result=model.predict(new_data_scaled)
        return render_template("home.html", result=result)

    else:
        return render_template("home.html")
    

# def prediction():
#     pass





if __name__== "__main__" :
    app.run(host="0.0.0.0")
