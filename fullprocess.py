import os
import pandas as pd
import json
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting

with open('config.json','r') as f:
    config = json.load(f)
input_folder_path = config['input_folder_path']
data_path = os.path.join(config['output_folder_path'], config['ingested_data_file'])
model_path = os.path.join(config['output_model_path'], config['model'])
deployed_model_path = os.path.join(config['prod_deployment_path'], config['model'])
test_records = os.path.join(config['test_records'])
deployed_test_records = os.path.join(config['prod_deployment_path'],
                            config['test_records'])
output_path = os.path.join(config['classification_report'])

ingested_files = pd.read_csv('ingested_files.txt', sep=' ', header=None)
unique_ingested_files = set(ingested_files[1].unique())
non_processed_files = set(os.listdir(input_folder_path)).difference(unique_ingested_files)

if non_processed_files:
    ingestion.merge_multiple_dataframe()

    with open(deployed_test_records, 'r') as file:
        f1 = file.read().split(' ')[-2]

    scoring.score_model(deployed_model_path, data_path, test_records)
    with open(test_records, 'r') as file:
        new_f1 = file.read().split(' ')[-2]
    if new_f1 < f1:
        print('new_f1 > f1')
        training.train_model(data_path, model_path)
        deployment.move_artifacts_to_production_deployment_dir()
        y_test, y_preds = reporting.score_model()
        reporting.classification_report_image(y_test, y_preds, output_path)
        y_preds = diagnostics.model_predictions()
        print('y_preds:')
        print(y_preds)
        summary =  diagnostics.dataframe_summary()
        print('dataframe summary:')
        print(summary)
        info =  diagnostics.dataframe_info()
        time =  diagnostics.execution_time()
        print('execution time for ingestion and training:')
        print(time)
        diagnostics.outdated_packages_list()







