import os
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

from src.logger.logging import logging
from src.exception.exception import CustomeException

def SaveObject(file_path, obj):
    try:
        dirPath = os.path.dirname(file_path)

        os.makedirs(dirPath, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomeException(e, sys)
    

def EvaluateModel(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomeException(e,sys)
    
def LoadObject(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in LoadObject function utils')
        raise CustomeException(e,sys)
