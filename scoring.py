import pandas as pd
import os
from sklearn.metrics import f1_score
import joblib
import json
from datetime import datetime

with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'], config['model'])
test_data_path = os.path.join(config['test_data_path'], config['test_data_file'])
test_records = os.path.join(config['test_records'])

def score_model(model_path, test_data_path, test_records):
    model = joblib.load(model_path)
    test_data = pd.read_csv(test_data_path)
    X_test = test_data[['lastmonth_activity','lastyear_activity','number_of_employees']]
    y_test = test_data['exited']
    y_preds = model.predict(X_test)
    f1 = f1_score(y_test, y_preds)

    dateTimeObj=datetime.now()
    thetimenow=str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)
    allrecords=[model_path, test_data_path,len(test_data), thetimenow, f1]
    with open(test_records,'a') as MyFile:
        for element in allrecords:
            MyFile.write(str(element) + " ")
        MyFile.write('\n')

if __name__ == '__main__':
    core_model(model_path, test_data_path, test_records)