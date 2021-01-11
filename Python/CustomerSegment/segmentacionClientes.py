#IMPORTAR LIBRERÍAS


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import collections
import warnings
import os
df = {}

def primer_vistazo_datos():
    print(df.head())
    print("\n\n")
    print(df.shape)
    print("\n\n")
    print(df.describe())
    print("\n\n")
    print(df.dtypes)
    print("\n\n")
    print(df.isnull().sum())

def visualizacion_datos():
    plt.style.use('fivethirtyeight')
    histogramas()

def histogramas():
    #visualizamos histogramas de cada característica
    plt.figure(1 , figsize = (17 , 8))
    n = 0 
    for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(1 , 3 , n)
        plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
        sns.histplot(df[x])
        plt.title('Histograma de {}'.format(x))
    plt.show()

def datosxgenero():
    plt.figure(1 , figsize = (17 , 8))
    sns.countplot(y = 'Gender' , data = df)
    plt.show()

def relacion_variables():
    #relacion entre todas las caracteristicas
    plt.figure(1 , figsize = (17 , 8))
    n = 0 
    for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
            n += 1
            plt.subplot(3 , 3 , n)
            plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
            sns.regplot(x = x , y = y , data = df)
            plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
    plt.show()

    #relacion entre la edad y los ingresos anuales
    plt.figure(1 , figsize = (17 , 8))
    for gender in ['Male' , 'Female']:
        plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = df[df['Gender'] == gender] ,
                    s = 200 , alpha = 0.5 , label = gender)
    plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 
    plt.title('Edad vs Renta anual según Género')
    plt.legend()
    plt.show()


    #relacion entre los ingresos y lo gastado
    plt.figure(1 , figsize = (17 , 8))
    for gender in ['Male' , 'Female']:
        plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,
                    data = df[df['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)
    plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 
    plt.title('Renta anual vs Puntuación gasto, según Género')
    plt.legend()
    plt.show()

def distribucionxgenero():
    plt.figure(1 , figsize = (17 , 8))
    n = 0 
    for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1 
        plt.subplot(1 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.violinplot(x = cols , y = 'Gender' , data = df , palette = 'vlag')
        sns.swarmplot(x = cols , y = 'Gender' , data = df)
        plt.ylabel('Gender' if n == 1 else '')
        plt.title('Distribución según género' if n == 2 else '')
    plt.show()

def clusteringKmeans(numclusters,maxiter,initialclusters,data):
    
    
    algorithm = (KMeans(n_clusters = numclusters ,init='k-means++', n_init = initialclusters ,max_iter=maxiter, 
                        tol=0.0001, algorithm='elkan') )
    algorithm.fit(data)

    return algorithm;




#EXPLORACIÓN DATOS

df = pd.read_csv(r'./input/Mall_Customers.csv')



primer_vistazo_datos()

#VISUALIZACIÓN DATOS

plt.style.use('fivethirtyeight')

#histogramas

histogramas()

#datos según género

datosxgenero()


#relación entre variables, visualización todas juntas (matriz correlación) y por parejas individualmente

relacion_variables()



#distribución de los valores de las características en función de si es mujer o hombre

distribucionxgenero()


#CLUSTERING

#segmentación usando edad y cuanto gasta

'''Edad y Puntuación según gasto'''
X1 = df[['Age' , 'Spending Score (1-100)']].iloc[: , :].values



numMaxClusters = 15
numIter = 400
nClustersInic = 10





inertiavec = []
for n in range(1 , numMaxClusters):
    algorit= clusteringKmeans(n,numIter,nClustersInic,X1)
    inertiavec.append(algorit.inertia_)



plt.figure(1, figsize = (17 , 8))
plt.plot(np.arange(1 , numMaxClusters) , inertiavec , '-' , alpha = 0.5)
plt.xlabel('Nº clusters') , plt.ylabel('Inertia')
plt.show()



algorit = clusteringKmeans(4,numIter,nClustersInic,X1)
labels1 = algorit.labels_
centroids1 = algorit.cluster_centers_

#PLOTEO DATOS -> GASTO - EDAD

#usamos la h para el intervalo entre maximoo y minimo a la hora de hacer la matriz, parecido a linspace en matlab
h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
#creamos la matriz para representar
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = algorit.predict(np.c_[xx.ravel(), yy.ravel()]) 


plt.figure(1, figsize = (17 , 8))
#plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap = plt.cm.Pastel1, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = df , c = labels1 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1]  , c = 'red', alpha = 0.5)
plt.ylabel('Puntuación según gasto') , plt.xlabel('Edad')
plt.show()



#segmentación usando ingresos anuales y cuanto gasta

X2 = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values
inertiavec = []
for n in range(1 , numMaxClusters):
    algorit= clusteringKmeans(n,numIter,nClustersInic,X2)
    inertiavec.append(algorit.inertia_)



plt.figure(1 , figsize = (17 , 8))
plt.plot(np.arange(1 , numMaxClusters) , inertiavec , '-' )
plt.xlabel('Nº clusters') , plt.ylabel('Inertia')
plt.show()


algorit = clusteringKmeans(5,numIter,nClustersInic,X2)
labels2 = algorit.labels_
centroids2 = algorit.cluster_centers_

#PLOTEO DATOS -> GASTO - INGRESO

h = 0.01
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z2 = algorit.predict(np.c_[xx.ravel(), yy.ravel()])



plt.figure(1 , figsize = (17 , 8))
plt.clf()
Z2 = Z2.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap = plt.cm.Pastel1, aspect = 'auto', origin='lower')

plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = df , c = labels2 )
plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , c = 'red' )
plt.ylabel('Puntuación gasto') , plt.xlabel('Ingreso anual (k$)')
plt.show()

#segmentación usando edad, ingresos anuales y cuanto gasta


X3 = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values

inertiavec = []
for n in range(1 , numMaxClusters):
    algorit= clusteringKmeans(n,numIter,nClustersInic,X3)
    inertiavec.append(algorit.inertia_)


plt.figure(1 , figsize = (17 , 8))
plt.plot(np.arange(1 , numMaxClusters) , inertiavec , '-' )
plt.xlabel('Nº Clusters') , plt.ylabel('Inertia')
plt.show()

algorit = clusteringKmeans(6,numIter,nClustersInic,X3)
classes = algorit.fit_predict(X3)
labels3 = algorit.labels_
centroids3 = algorit.cluster_centers_




df['label3'] =  labels3
trace1 = go.Scatter3d(
    x= df['Age'],
    y= df['Spending Score (1-100)'],
    z= df['Annual Income (k$)'],
    mode='markers',
     marker=dict(
        color = df['label3'], 
        size= 20,
        line=dict(
            color= df['label3'],
            width= 12
        ),
        opacity=0.8
     )
)
data = [trace1]
layout = go.Layout(
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0
#     )
    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'Edad'),
            yaxis = dict(title  = 'Puntuación según gasto'),
            zaxis = dict(title  = 'Ingreso anual')
        )
)
fig = go.Figure(data=data, layout=layout)
fig.show()
print(collections.Counter(classes))
plt.figure(1,figsize = (15 , 7))
plt.hist(classes)
plt.show()
#py.offline.iplot(fig)







