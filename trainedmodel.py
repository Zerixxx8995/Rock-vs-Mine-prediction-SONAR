#Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data collection and processing
#loading the dataset to pandas Dataframe

sonar_data = pd.read_csv('C:/Users/manav/Desktop/ML/Rock-vs-Mine-prediction-SONAR/Copy of sonar data.csv', header=None)
#as columns have no name

# print(sonar_data.head())
#prints first 5 rows of data set

#number of rows and columns
# print(sonar_data.shape)

# print(sonar_data.describe())
 #statistical measures of the data

# print(sonar_data[60].value_counts())  
#as rock and mine info is in 60th column and almost equal number for both thus prediction would be good


#M == mine R == rock , More the data more accurate the assumption

# print(sonar_data.groupby(60).mean())

#seperating data and labels
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]

# print(X)
# print(Y)

#Training and test data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1) 
#0.1 means 10 percent for test,stratify Y as data needs to be splitted based on rock and mine,

# print(X.shape,X_train.shape,X_test.shape)

# print(X_train)
# print(Y_train)

model=LogisticRegression()

#training the Logistic Regression model with training data
model.fit(X_train,Y_train)
