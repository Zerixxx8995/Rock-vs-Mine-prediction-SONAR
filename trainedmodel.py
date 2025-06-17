#Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data collection and processing
#loading the dataset to pandas Dataframe

sonar_data = pd.read_csv('C:/Users/manav/Desktop/ML/Rock-vs-Mine-prediction-SONAR/Copy of sonar data.csv', header=None)

print(sonar_data.head())