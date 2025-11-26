import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

X_train=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/X_train.csv', index_col=0)
X_test=pd.read_csv('/home/ubuntu/examen-dvc/data/processed_data/X_test.csv', index_col=0)


scaler=StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled= X_test.copy()
nums = X_train.select_dtypes(include=['number']).columns.tolist()
X_train_scaled[nums]=scaler.fit_transform(X_train_scaled[nums])
X_test_scaled[nums]=scaler.transform(X_test_scaled[nums])

X_train_scaled.to_csv('/home/ubuntu/examen-dvc/data/processed_data/X_train_scaled.csv')
X_test_scaled.to_csv('/home/ubuntu/examen-dvc/data/processed_data/X_test_scaled.csv')