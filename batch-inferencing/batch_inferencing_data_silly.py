
import os
import numpy as np
from azureml.core import Model
import joblib
import time
import pandas as pd


def init():
    # Runs when the pipeline step is initialized
    global model

    # load the model
    print('****loaded model**********')
    model_path = Model.get_model_path('email_classifier')
    model = joblib.load(model_path)


def run(mini_batch):
    # This runs for each batch
    print(f'run method start: {__file__}, run({mini_batch})')
    resultList = []
    all_predictions = pd.DataFrame()
    
    for idx, file_path in enumerate(mini_batch):
        file_name, file_extension = os.path.splitext(os.path.basename(file_path))
       
        #print(file_path)
        #data = pd.read_csv(file_path)
        
        text_file = open(file_path, "r")
        data = text_file.read()
        text_file.close()
        result = model.predict([data])
        print(data)
        resultList.append("{}: {}".format(os.path.basename(file_path), result[0]))
    #return resultList
        
        #for _, row in result_df.iterrows():
        #    result_list.append((row))


    #Return all rows formatted as a Pandas dataframe
    return pd.DataFrame(resultList)

