import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])


##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    predictions = np.array(model_predictions(test_data_path+'/testdata.csv'))
    actual = np.array(pd.read_csv(test_data_path+'/testdata.csv')['exited'])
    confusion_matrix = metrics.confusion_matrix(actual, predictions)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.savefig(dataset_csv_path+'/confusionmatrix.png')


if __name__ == '__main__':
    score_model()
