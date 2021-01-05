
##--- functions
def datetounix(df):
    # Initialising unixtime list
    unixtime = []
    
    # Running a loop to convert Date to seconds
    for date in df['datetime']:
        unixtime.append(time.mktime(date.timetuple()))
    
    # Replacing Date with unixtime list
    df['datetime'] = unixtime
    return(df)

#--import libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.ensemble import ExtraTreesClassifier
import operator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense


# read train dataframe
# file_path = os.path.join(os.path.abspath(''), 'train.csv')
df_train = pd.read_csv("train.csv", encoding='ISO-8859-1', engine='c')

del(df_train['var1'])
del(df_train['var2'])
# read test dataframe
# file_path = os.path.join(os.path.abspath(''), 'test.csv')
df_test = pd.read_csv("test.csv", encoding='ISO-8859-1', engine='c')
#df_train.info()
del(df_test['var1'])
del(df_test['var2'])
#--data cleaning

df_train['datetime'] = pd.to_datetime(df_train['datetime'])
df_test['datetime'] = pd.to_datetime(df_test['datetime'])
#df_test.info()

#-create external features from datetime

df_test['Weekday'] = [datetime.weekday(date) for date in df_test.datetime]
df_test['Year'] = [date.year for date in df_test.datetime]
df_test['Month'] = [date.month for date in df_test.datetime]
df_test['Day'] = [date.day for date in df_test.datetime]
df_test['Time'] = [((date.hour*60+(date.minute))*60)+date.second for date in df_test.datetime]
df_test['Week'] = [date.week for date in df_test.datetime]
df_test['Quarter'] = [date.quarter for date in df_test.datetime]

df_train['Weekday'] = [datetime.weekday(date) for date in df_train.datetime]
df_train['Year'] = [date.year for date in df_train.datetime]
df_train['Month'] = [date.month for date in df_train.datetime]
df_train['Day'] = [date.day for date in df_train.datetime]
df_train['Time'] = [((date.hour*60+(date.minute))*60)+date.second for date in df_train.datetime]
df_train['Week'] = [date.week for date in df_train.datetime]
df_train['Quarter'] = [date.quarter for date in df_train.datetime]

#--data preparation

# Create Dummy Variables for Train set
#df_train.loc[df_train.var2 == 'A', 'var2A'] = 1
#df_train.loc[df_train.var2 == 'B', 'var2B'] = 1

#df_train['var2A'].fillna(0, inplace=True)
#df_train['var2B'].fillna(0, inplace=True)

#df_train.drop(['var2'], axis=1, inplace=True)

# Create Dummy Variables for Test set
#df_test.loc[df_test.var2 == 'A', 'var2A'] = 1
#df_test.loc[df_test.var2 == 'B', 'var2B'] = 1

#df_test['var2A'].fillna(0, inplace=True)
#df_test['var2B'].fillna(0, inplace=True)


print("ANTES DATETOUNIX \n")
# Creating X_test
X_test = datetounix(df_test).drop(['ID'], axis=1).values

# Remove target column from the df
df_train_features = df_train.drop(['electricity_consumption', 'ID'], axis=1)

# Convet timestamp to seconds
df_train_features = datetounix(df_train_features)

# store features in X array
X = df_train_features.values
y = df_train['electricity_consumption'].values

#--visualisation of features
print("ANTES TREESCLASSIFIER \n")
# create an instance for tree feature selection
tree_clf = ExtraTreesClassifier()

# fit the model
#tree_clf.fit(X, y)

# Preparing variables
#importances = tree_clf.feature_importances_
#feature_names = df_train_features.columns.tolist()

#feature_imp_dict = dict(zip(feature_names, importances))
#sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)

#indices = np.argsort(importances)[::-1]

# Print the feature ranking
#print("Feature ranking:")

#for f in range(X.shape[1]):
#    print("feature %d : %s (%f)" % (indices[f], sorted_features[f][0], sorted_features[f][1]))

# Plot the feature importances of the forest
#plt.figure(0)
#plt.title("Feature importances")
#plt.bar(range(X.shape[1]), importances[indices],
#       color="r", align="center")
#plt.xticks(range(X.shape[1]), indices)
#plt.xlim([-1, X.shape[1]])
#plt.show()

############ Data Scaling ###################
print("ANTES DATASCALING \n")
sc = StandardScaler()
X = sc.fit_transform(X)

#X_test = sc.transform(X_test)
print("--------"+str(len(X))+"---------")
P = int(round(len(X)*0.8))
X_train, X_test = X[:P],X[(P+1):]
Y_train, Y_test = y[:P],y[(P+1):]

np.savetxt('xtrain.txt',X_train)
np.savetxt('xtest.txt',X_test)
np.savetxt('ytrain.txt',Y_train)
np.savetxt('ytest.txt',Y_test)
miny = min(Y_train)
maxy = max(Y_train)


y_norm = (Y_train - miny)/(maxy - miny)
#y_norm
print("DESPUÉS DATASCALING \n")
#--IMPLEMENTACIÓN RED


print("ANTES ANN \n")
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_norm, batch_size = 10, epochs = 100)
print("DESPUES DATASCALING \n")
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)



y_pred = (y_pred * (maxy - miny)) + miny

np.savetxt("yout.txt",y_pred)

predictions = [int(elem) for elem in y_pred]

df_solution = pd.DataFrame()
df_solution['ID'] = df_test.ID

# Prepare Solution dataframe
df_solution['electricity_consumption'] = predictions
df_solution['electricity_consumption'].unique()

print("HOLA")
df_solution