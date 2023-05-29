from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.getcwd()+'/'+config['output_model_path']+'/'
dataset_csv_path = os.getcwd()+'/'+config['output_folder_path']+'/'
prod_deployment_path = os.getcwd()+'/'+config['prod_deployment_path']+'/'


####################function for deployment
def copy_to_deployment():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    os.makedirs(os.path.dirname(prod_deployment_path), exist_ok=True)
    shutil.copy(model_path+'trainedmodel.pkl', prod_deployment_path+'trainedmodel.pkl')
    shutil.copy(model_path+'latestscore.txt', prod_deployment_path+'latestscore.txt')
    shutil.copy(dataset_csv_path+'ingestedfiles.txt', prod_deployment_path+'ingestedfiles.txt')

if __name__ == '__main__':
    copy_to_deployment()

