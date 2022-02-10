
import os
import numpy as np
from azureml.core import Model
import joblib
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
    print('type of mini batch')
    print(str(type(mini_batch)))
    # process each file in the batch
    for f in mini_batch:
        print('****working on mini_batch**********')
        print(f)
        #open text file in read mode
        text_file = open(f, "r")
        data = text_file.read()
        text_file.close()
        result = model.predict([data])
        print(data)
        resultList.append("{}: {}".format(os.path.basename(f), result[0]))
    return resultList
