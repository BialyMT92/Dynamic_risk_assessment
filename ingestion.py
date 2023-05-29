import errno

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.getcwd()+'/'+config['input_folder_path']+'/'
output_folder_path = os.getcwd()+'/'+config['output_folder_path']+'/'


#############Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    final_dataframe = pd.DataFrame(columns=['corporation', 'lastmonth_activity', 'lastyear_activity',
                                            'number_of_employees', 'exited'])

    filenames = os.listdir(input_folder_path)

    for each_filename in filenames:
        currentdf = pd.read_csv(input_folder_path+each_filename, index_col=False)
        final_dataframe = final_dataframe.append(currentdf).reset_index(drop=True)
    final_dataframe.drop_duplicates(inplace=True)

    if not os.path.exists(output_folder_path):
        try:
            os.makedirs(os.path.dirname(output_folder_path))
        except OSError as exc: #Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    final_dataframe.to_csv(output_folder_path+'finaldata.csv')

    with open(output_folder_path+'ingestedfiles.txt', 'w') as file:
        file.write('\n'.join(filenames))


if __name__ == '__main__':
    merge_multiple_dataframe()
