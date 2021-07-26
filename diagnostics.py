
import pandas as pd
import numpy as np
import timeit
import os
import json
import joblib
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'], config['ingested_data_file'])
model_path = os.path.join(config['prod_deployment_path'], config['model'])
test_data_path = os.path.join(config['test_data_path'], config['test_data_file'])
test_records = os.path.join(config['test_records'])

##################Function to get model predictions
def model_predictions():
    model = joblib.load(model_path)
    test_data = pd.read_csv(test_data_path)
    X_test = test_data[['lastmonth_activity','lastyear_activity','number_of_employees']]
    # y_test = test_data['exited']
    y_preds = model.predict(X_test)
    return y_preds

##################Function to get summary statistics
def dataframe_summary():
    df = pd.read_csv(dataset_csv_path)
    return df.describe()

def dataframe_info():
    df = pd.read_csv(dataset_csv_path)
    return df.info()

def ingestion_timing():
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    timing=timeit.default_timer() - starttime
    return timing

def training_timing():
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    timing=timeit.default_timer() - starttime
    return timing

def execution_time():
    ingestion_timings=[]
    training_timings=[]

    for idx in range(1):
        ingestion_timings.append(ingestion_timing())
        training_timings.append(training_timing())

    final_output=[]
    final_output.append(np.mean(ingestion_timings))
    # final_output.append(np.std(ingestion_timings))
    # final_output.append(np.min(ingestion_timings))
    # final_output.append(np.max(ingestion_timings))
    final_output.append(np.mean(training_timings))
    # final_output.append(np.std(training_timings))
    # final_output.append(np.min(training_timings))
    # final_output.append(np.max(training_timings))
    return final_output

##################Function to check dependencies
def outdated_packages_list():

    outdated = subprocess.check_output(['pip', 'list','--outdated'])
    with open('outdated.txt', 'wb') as f:
        f.write(outdated)

if __name__ == '__main__':
    y_preds = model_predictions()
    print('y_preds:')
    print(y_preds)
    summary = dataframe_summary()
    print('dataframe summary:')
    print(summary)
    info = dataframe_info()
    print('dataframe info:')
    print(info)
    time = execution_time()
    print('execution time for ingestion and training:')
    print(time)
    outdated_packages_list()






