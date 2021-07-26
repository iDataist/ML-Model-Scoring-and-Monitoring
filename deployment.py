from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

ingestion_records = config['ingestion_records']
model_path = config['output_model_path']
model = config['model']
test_records = config['test_records']
prod_deployment_path = config['prod_deployment_path']

####################function for deployment
def move_artifacts_to_production_deployment_dir():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    os.rename(ingestion_records, os.path.join(prod_deployment_path, ingestion_records))
    os.rename(os.path.join(model_path, model), os.path.join(prod_deployment_path, model))
    os.rename(test_records, os.path.join(prod_deployment_path, test_records))

if __name__ == '__main__':
    move_artifacts_to_production_deployment_dir()
