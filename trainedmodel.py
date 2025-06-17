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

#accuracy on training data above 70 percent shd be good
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

# print('Accuracy on training data:',training_data_accuracy)

#accuracy on training data above 70 percent shd be good
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

# print('Accuracy on test data:',test_data_accuracy)


#Making a predictive system

input_data=(0.0286,0.0453,0.0277,0.0174,0.0384,0.099,0.1201,0.1833,0.2105,0.3039,0.2988,0.425,0.6343,0.8198,1,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.589,0.2872,0.2043,0.5782,0.5389,0.375,0.3411,0.5067,0.558,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.265,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062
)

# changing the list(input data) to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a Mine')