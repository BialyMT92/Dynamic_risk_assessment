import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import json
import os
import sys

with open('config.json', 'r') as f:
    config = json.load(f)

##################Check and read new data
#first, read ingestedfiles.txt
ingestedfile_loc = os.path.join(config['prod_deployment_path']+'/ingestedfiles.txt')
with open (ingestedfile_loc, 'r') as file:
    old_filenames = file.read().splitlines()

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_filenames = os.listdir(config['input_folder_path'])
differences = list(set(new_filenames) - set(old_filenames))

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not bool(differences):
    sys.exit("No new input data")
else:
    os.system('python3 ingestion.py')
##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
old_score_loc = os.path.join(config['prod_deployment_path']+'/latestscore.txt')
with open(old_score_loc, 'r') as file:
    old_score = file.read()
old_score = float(old_score)
os.system('python3 scoring.py')
new_score_loc = os.path.join(config['output_model_path']+'/latestscore.txt')
with open(new_score_loc, 'r') as file:
    new_score = file.read()
new_score = float(new_score)

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_score > old_score:
    sys.exit("No drift detected")
else:
    os.system('python3 training.py')

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
os.system('python3 deployment.py')

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system('python3 diagnostics.py')
os.system('python3 reporting.py')






