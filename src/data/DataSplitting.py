import pandas as pd
import numpy as np
#import sklearn
from sklearn.model_selection import train_test_split

df_raw=pd.read_csv('/home/ubuntu/examen-dvc/data/raw_data/raw.csv')

y=df_raw['silica_concentrate']
X=df_raw.drop(['silica_concentrate','date'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

X_train.to_csv('/home/ubuntu/examen-dvc/data/processed_data/X_train.csv')
X_test.to_csv('/home/ubuntu/examen-dvc/data/processed_data/X_test.csv')
y_train.to_csv('/home/ubuntu/examen-dvc/data/processed_data/y_train.csv')
y_test.to_csv('/home/ubuntu/examen-dvc/data/processed_data/y_test.csv')
