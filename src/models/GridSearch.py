import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pickle

X_train_scaled=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/X_train_scaled.csv', index_col=0)
y_train=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/y_train.csv', index_col=0)

parameters= {'alpha':[0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100]}
model=Ridge()

grid=GridSearchCV(estimator=model,param_grid=parameters,cv=5)
grid.fit(X_train_scaled, y_train)

best_alpha=grid.best_estimator_

print(best_alpha)
with open("/home/ubuntu/examen-dvc/models/best_alpha.pkl", "wb") as f:
    pickle.dump(best_alpha, f)