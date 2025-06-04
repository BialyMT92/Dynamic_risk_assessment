from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics
import json
import os
import scoring


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = ## your api key

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    data_location = request.args.get('data_location')
    predition = diagnostics.model_predictions(data_location)
    #call the prediction function you created in Step 3
    return str(predition)

#######################Scoring Endpoint
@app.route("/scoring", methods=['POST','OPTIONS'])
def score():
    model_location = request.args.get('model_location')
    f1score = scoring.score_model(model_location)
    #check the score of the deployed model
    return f1score

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary():
    stats = diagnostics.dataframe_summary()
    #check means, medians, and modes for each column
    return str(stats)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag():
    list = []
    exe_time = diagnostics.execution_time()
    depend = diagnostics.outdated_packages_list()
    nan = diagnostics.nan_values()
    list.append(exe_time)
    list.append(depend)
    list.append(nan)
    return str(list)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
