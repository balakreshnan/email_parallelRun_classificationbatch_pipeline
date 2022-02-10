
import argparse
from azureml.core import Run

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def summarize_classification(y_test, y_pred, run):
    acc = accuracy_score(y_test, y_pred, normalize=True) #how many predictions correct %
    num_acc = accuracy_score(y_test, y_pred, normalize = False)
    prec = precision_score(y_test, y_pred, average = 'weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    run.log('acc count', num_acc)
    run.log('Accuracy', acc)
    run.log('prec', prec)
    run.log('recall', recall)
    
    print('aacuracy count:', num_acc)
    print('accuracy score:', acc)
    print('precision:', prec)
    print('recall:', recall)
    


def getRuntimeArgs():
    # Get script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, dest='training_dataset_id', help='training dataset')
    args = parser.parse_args()

    return args

def model_train(ds_df, run):
    
    X = ds_df['text']
    Y = ds_df['labels']
    #sklearn pipeline
    clf = Pipeline([
                            ('count_vectorizer', CountVectorizer()),
                            ('classifier', LogisticRegression(solver='lbfgs', max_iter=10000))
                        ])
    #output of convectorizer, feed to classifier
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    print('type of x_test')
    print(type(x_test))
    model = clf.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print('*************************')
    print('model predictions:')
    print(y_pred)
    summarize_classification(y_test, y_pred, run)

    return model



def main():
    args = getRuntimeArgs()
    
    # Get the experiment run context
    run = Run.get_context()
    
    dataset_dir = './dataset/'
    os.makedirs(dataset_dir, exist_ok=True)
    ws = run.experiment.workspace
    print(ws)
    
    
    print("Loading Data...")
    data = run.input_datasets['training_data'].to_pandas_dataframe()
    
    
    print(data.columns)
    lr = model_train(data, run)
    
    
    # Save the trained model
    model_file = 'email_classifier.pkl'
    joblib.dump(value=lr, filename=model_file)
    run.upload_file(name = 'outputs/' + model_file, path_or_stream = './' + model_file)

    # Complete the run
    run.complete()


    # Register the model
    run.register_model(model_path='outputs/email_classifier.pkl', model_name='email_classifier',
                       tags={'Training context':'spam or ham'})

    #print('Model trained and registered.')
 

if __name__ == "__main__":
    main()
