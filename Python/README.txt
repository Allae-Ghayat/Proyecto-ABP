INSTRUCCIONES DE EJECUCIÓN

Todos los códigos del proyecto han sido realizados en lenguaje Python.
Las bibliotecas necesarias para la ejecución de los códigos se encuentran en el archivo requirements.txt

- Predicción consumo de electricidad

Una vez estén las librerias instaladas, tan solo haciendo python electricidad.py, 
la red procederá a realizar el entrenamiento y luego a predecir con los datos de test.
En todo lo anterior la ejecución tarda varios minutos.
Finalmente se generarán gráficas donde se podrán comparar los consumos reales de los días de test.csv con con nuestras predicciones.
Estas gráficas por una parte serán de cada mes de 2015 para mejor apreciación de las diferencias y además
generamos una gráfica de cada año.


- Segmentación de clientes

Para la segmentación de clientes se encuentra todo en el archivo segmentacionClientes.py. Los datasets deben estar dentro de una carpeta input en el mismo lugar dónde se encuentre el
archivo segmentacionClientes.py . 
Dentro del archivo segmentacionClientes.py se encuentra divido en apartados diferenciados por líneas de comentarios las fases del proyecto. Inicialmente está el estudio
sobre los datos, estos se encuentran separados en funciones, al igual que el algoritmo de clustering. Y después del estudio de datos están las 3 formas de clustering que hemos usado,
siendo la última la finalmente presentada al cliente (usando edad, renta anual y puntuación según gasto). Las dos últimas gráficas que indican el número de clientes por grupo,
y la visualización 3D del resultado del clustering, se han realizado con la biblioteca plotly la cuál abrirá una página en el navegador que tenga instalado para mostrar la gráfica
con la que se puede interactuar igual que haciamos con las gráficas de Matlab.