import pickle
import subprocess

import pandas as pd
import numpy as np
import timeit
import os
import json

##################Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(input_data_path):
    #read the deployed model and a test dataset, calculate predictions
    with open (model_path+'/trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    testdata = pd.read_csv(input_data_path)

    X = testdata.loc[:, ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = testdata['exited'].values.reshape(-1, 1).ravel()

    predicted = model.predict(X)
    return predicted

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here means, medians, std for columns and nas
    thedata = pd.read_csv(dataset_csv_path+'/finaldata.csv')
    means = thedata.mean(numeric_only=True)
    medians = thedata.median(axis=0, numeric_only=True)
    std = thedata.std(numeric_only=True)

    statistics = [means, medians, std]
    return statistics

def nan_values():
    thedata = pd.read_csv(dataset_csv_path+'/finaldata.csv')

    nas = list(thedata.isna().sum())
    nascompar = [nas[i]/len(thedata.index) for i in range(len(nas))]
    return nascompar
##################Function to get timings
def execution_time():
    times = []
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    totaltime = timeit.default_timer() - starttime
    times.append(totaltime)

    starttime = timeit.default_timer()
    os.system('python3 training.py')
    totaltime = timeit.default_timer() - starttime
    times.append(totaltime)

    return times

##################Function to check dependencies
def outdated_packages_list():
    #get a list of
    list_of_dep = subprocess.check_output(['pip', 'list', '--outdated'])
    return list_of_dep

if __name__ == '__main__':
    print(model_predictions(dataset_csv_path + '/finaldata.csv'))
    print(dataframe_summary())
    print(nan_values())
    print(execution_time())
    print(outdated_packages_list())






    
