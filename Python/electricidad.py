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
import random
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

del(df_test['electricity_consumption']) 
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


# Creating X_test
aux=df_test.copy()
X_test = datetounix(aux).drop(['ID'], axis=1).values

# Remove target column from the df
df_train_features = df_train.drop(['electricity_consumption', 'ID'], axis=1)

# Convet timestamp to seconds
df_train_features = datetounix(df_train_features)

# store features in X array
X_train = df_train_features.values
y_train = df_train['electricity_consumption'].values


############ Data Scaling ###################

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

miny = min(y_train)
maxy = max(y_train)

y_normtrain = (y_train - miny)/(maxy - miny)

# Initialising the ANN
def baseline_model():
	classifier = Sequential()
	# Adding the input layer and the first hidden layer
	classifier.add(Dense(11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

	classifier.add(Dense(5, kernel_initializer='uniform', activation='relu'))
	# Adding the output layer
	classifier.add(Dense(1, kernel_initializer = 'uniform',activation='sigmoid'))

	# Compiling the ANN
	classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')

	return classifier


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

estimator=KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=10,verbose=0)

# Fitting the ANN to the training set

kfold = KFold(n_splits=3)

results = cross_val_score(estimator, X_train, y_normtrain, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
print("---Training finished---")
# Part 3 - Making the predictions and evaluating the model

estimator.fit(X_train, y_normtrain)
prediction = estimator.predict(X_test)

print("prediction=")
print(prediction)

#accuracy_score(Y_test, prediction)


preddesnorm = (prediction * (maxy - miny)) + miny
print("pred desnorm=")
print(preddesnorm)


mybdtest=df_test.copy()
mybdtest.insert(12,'electricity_consumption',np.round(preddesnorm,2))

myfullbd=pd.concat([df_train,mybdtest],ignore_index=True)
myfullbd.sort_values(by='datetime')



fullbd=read_csv('fullbd.csv')


fullrealconsums=fullbd['electricity_consumption'].values
predconsums=myfullbd['electricity_consumption'].values

#Con esto genero comparaciones graficas para ver como ha predecido 
#los consumos de cada hora de los ultimos 7-8 dias de cada mes de este año

"""
year=2015

for mes in range(1,13):

		realyear=fullbd.loc[(fullbd['Year']==year)  & (fullbd['Month']==mes) & (fullbd['Day']>23)  ]['electricity_consumption' ].values
		predyear=myfullbd.loc[(myfullbd['Year']==year) & (fullbd['Month']==mes) & (myfullbd['Day'] > 23) ]['electricity_consumption'].values

		if (len(realyear)>0) & (len(predyear)>0):
			fig,ax=plt.subplots(2)

			plt.subplots_adjust(hspace=0.4)

			ax[0].plot(realyear)
			ax[0].set_title('Consumos de electricidad reales de %i, mes %i' % (year,mes))
			ax[1].set_title('Consumos de electricidad predecidos de %i, mes %i' % (year,mes))
			ax[1].plot(predyear)
			plt.savefig('comp%i%i.png' % (year,mes))
			plt.close("all")
"""

#Para generar graficas anuales
"""
for year in range(2013,2018):

		realyear=fullbd.loc[(fullbd['Year']==year) & (fullbd['Day']>23)  ]['electricity_consumption' ].values
		predyear=myfullbd.loc[(myfullbd['Year']==year) & (myfullbd['Day'] > 23) ]['electricity_consumption'].values

		if len(realyear)>0 and len(predyear)>0:
			fig,ax=plt.subplots(2)

			plt.subplots_adjust(hspace=0.4)

			ax[0].plot(realyear)
			ax[0].set_title('Consumos reales de %i' % year)
			ax[1].set_title('Consumos predecidos de %i ' % year)
			ax[1].plot(predyear)
			plt.savefig('comp%i.png' % (year))
			plt.close("all")
"""
