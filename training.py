import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
import json
import joblib

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'], config['model'])
dataset_csv_path = os.path.join(config['output_folder_path'], config['ingested_data_file'])

df = pd.read_csv(dataset_csv_path)
X = df[['lastmonth_activity','lastyear_activity','number_of_employees']]
y = df['exited']
#################Function for training the model
def train_model():

    #use this logistic regression for training
    lr = LogisticRegression()

    #fit the logistic regression to your data
    lr.fit(X, y)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    joblib.dump(lr, model_path)

if __name__ == '__main__':
    train_model()
