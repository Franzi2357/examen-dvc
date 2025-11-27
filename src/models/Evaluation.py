import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import pickle
import json



X_train_scaled=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/X_train_scaled.csv', index_col=0)
y_train=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/y_train.csv', index_col=0)
X_test_scaled=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/X_test_scaled.csv', index_col=0)
y_test=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/y_test.csv', index_col=0)
X_train=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/X_train.csv', index_col=0)
X_test=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/X_test.csv', index_col=0)

with open("/home/ubuntu/examen-dvc/models/trained_model.pkl", "rb") as f:
    trained_model = pickle.load(f)

pred_train=trained_model.predict(X_train_scaled)
pred_test=trained_model.predict(X_test_scaled)

scores = {
    "Train Score":trained_model.score(X_train_scaled,y_train),
    "Test Score":trained_model.score(X_test_scaled,y_test),
    "Train MAE": mean_absolute_error(y_train, pred_train),
    "Test MAE": mean_absolute_error(y_test, pred_test),
    "Train MSE": mean_squared_error(y_train, pred_train),
    "Test MSE": mean_squared_error(y_test, pred_test),
    "Train R2": r2_score(y_train, pred_train),
    "Test R2": r2_score(y_test, pred_test)
}

with open("/home/ubuntu/examen-dvc/metrics/scores.json", "w") as f:
    json.dump(scores, f,indent=4)

Train_predict=X_train
Train_predict['Prediction']=pred_train
Test_predict=X_test
Test_predict['Prediction']=pred_test

Train_predict.to_csv('/home/ubuntu/examen-dvc/data/Prediction_Trainset.csv')
Test_predict.to_csv('/home/ubuntu/examen-dvc/data/Prediction_Testset.csv')