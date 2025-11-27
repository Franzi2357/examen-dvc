import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import pickle


X_train_scaled=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/X_train_scaled.csv', index_col=0)
y_train=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/y_train.csv', index_col=0)

with open("/home/ubuntu/examen-dvc/models/best_alpha.pkl", "rb") as f:
    best_alpha = pickle.load(f)

model=Ridge(best_alpha.alpha)
model.fit(X_train_scaled,y_train)

with open("/home/ubuntu/examen-dvc/models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)