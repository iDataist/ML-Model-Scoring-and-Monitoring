import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
ingested_data_file = config['ingested_data_file']
ingestion_records = os.path.join(config['ingestion_records'])

def merge_multiple_dataframe():
    df_list = []
    for file in os.listdir(input_folder_path):
        filepath = os.path.join(input_folder_path, file)
        temp_df = pd.read_csv(filepath)
        df_list.append(temp_df)

        dateTimeObj=datetime.now()
        thetimenow=str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)
        allrecords=[input_folder_path,file,len(temp_df),thetimenow]
        with open(ingestion_records,'a') as MyFile:
            for element in allrecords:
                MyFile.write(str(element) + " ")
            MyFile.write('\n')

    df = pd.concat(df_list, axis=0, ignore_index=True).drop_duplicates()
    df.to_csv(os.path.join(output_folder_path, ingested_data_file), index=False)

if __name__ == '__main__':
    merge_multiple_dataframe()
