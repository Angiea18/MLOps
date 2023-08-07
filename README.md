![Logo](https://blog.soyhenry.com/content/images/size/w2000/2022/04/Data2_logo.png)

# PROYECTO INDIVIDUAL Nº1 - DATA SCIENCE



## Machine Learning Operations (MLOps)

![MLOps](https://github.com/Angiea18/1-proyecto-individual-MLOps/blob/main/MLOps.png?raw=true)


**`by Angie Arango Zapata`**

## Contexto

El objetivo de este proyecto es desarrollar un proceso de **`Data Engineering`** para analizar un dataset de videojuegos y, utilizando técnicas de **`Machine Learning`**, crear un modelo de predicción. Además, se implementará una **`API`** que ofrecerá endpoints para acceder a los resultados del análisis y la predicción de precios.


## Dataset

El [dataset](https://github.com/Angiea18/1-proyecto-individual-MLOps/blob/main/steam_games.json) en cuestión posee información acerca de videojuegos y distintas características de estos. El mismo cuenta con 32135 filas (videojuego) y 16 columnas (características).


## Data Engineering y EDA

Para el trabajo de `Data Engineering`, no se pedia realizar tranformaciones específicas, sin embargo se llevó a cabo un proceso de transformación y preparación de los datos con el fin de leer correctamente el dataset. 

Para el `EDA` se solicito investigar las relaciones que hay entre las variables del dataset, ver si hay outliers o anomalías, y ver si hay algún patrón interesante que valga la pena explorar en un análisis posterior.

Esto consitió en:

1. Carga y preprocesamiento de datos:
- El script lee los datos del archivo `'steam_games.json'` y los convierte en un DataFrame de Pandas.
- La columna `'release_date'` se convierte en un tipo de dato datetime.
- La columna `'metascore'` se convierte en valores numéricos, ya que contenía puntajes en formato de texto.
- Se realizan cambios en la columna `'price'`:
    - Los valores que indican gratuidad se reemplazan con 0.
    - Los valores que comienzan con 'Starting at $' se reemplazan por el valor numérico correspondiente.
    - Los valores no numéricos se reemplazan por NaN.
- Se eliminan las filas que contienen valores faltantes en las columnas 'price' y 'metascore', creando un nuevo `DataFrame` [df2](https://github.com/Angiea18/1-proyecto-individual-MLOps/blob/main/df2.csv).
- Se extrae el año de la fecha en la columna 'release_date' y se agrega como una nueva columna `'year'` en el DataFrame df2.


2. Visualizaciones:
Se realizaron visualizaciones usando los datos del df2 exploratorias utilizando las librerías seaborn y matplotlib para comprender la distribución y relaciones entre las características del conjunto de datos:

- Se creó un *gráfico de barras* para visualizar el número de juegos lanzados por año.
- Se creó un *mapa de calor* para visualizar la matriz de correlación entre las características.
- Se generó un *diagrama de caja* para visualizar la distribución de los valores de 'price' y 'metascore'.
- Se creó un *gráfico de dispersión* para visualizar la relación entre el precio y el metascore.
- Se generó una *nube de palabras* a partir de los títulos de los juegos para visualizar las palabras más frecuentes en ellos.


Se pueden visualizar las transformaciones y los análisis realizados [aquí](https://github.com/Angiea18/1-proyecto-individual-MLOps/blob/main/steam_games_ML.ipynb)
## API

- Se solicitó efectuar la disponibilización de los siguientes endpoints a través del Framework **`FastAPI`**:.
- def genero( Año: str ): Se ingresa un año y devuelve un diccionario con los 5 géneros más repetidos en el orden correspondiente.

- def juegos( Año: str ): Se ingresa un año y devuelve una lista con los juegos lanzados en el año.

- def specs( Año: str ): Se ingresa un año y devuelve un diccionario con los 5 specs que más se repiten en el mismo en el orden correspondiente.

- def earlyaccess( Año: str ): Cantidad de juegos lanzados en un año con early access.

- def sentiment( Año: str ): Según el año de lanzamiento, se devuelve un diccionario con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento.

- def metascore( Año: str ): Top 5 juegos según año con mayor metascore.


El código para correr la API dentro de FastAPI se puede visualizar [aquí](https://github.com/Angiea18/1-proyecto-individual-MLOps/blob/main/main.py)

## Modelo de predicción - Machine Learning
Para el modelo de predicción **Machine Learning** se utilizó del dataset df2 las características **'genres'**, **'metascore'** y **'year'** para predecir el **`'price'`** de los videojuegos de Steam.

Creación del modelo:
- Se eliminaron las filas con valores faltantes en las columnas de interés ('genres', 'metascore', 'year').
- Se aplicó *one-hot encoding* para convertir las variables categóricas 'genres' en variables numéricas.
- Los datos se dividieron en conjuntos de entrenamiento y prueba.
- Se creó un modelo de regresión lineal múltiple y se utilizó como base para el modelo Bagging.
- Se entrenó el **`modelo Bagging`** utilizando el ensamble de 10 modelos de regresión lineal.

Evaluación del modelo:
- Se realizaron predicciones en el conjunto de prueba utilizando el modelo Bagging.
- Se calculó el **`RMSE`** utilizando las predicciones y los valores reales del precio de los videojuegos en el conjunto de prueba.
- Se guardó el modelo Bagging entrenado en un archivo llamado "modelo_bagging.pkl" utilizando la librería **`pickle`**.

API del modelo:

Se creó una **API** utilizando FastAPI para el despliegue del modelo y la consulta de predicciones del precio de los juegos de Steam.

- Se definió un Enum para los géneros disponibles en la API.
- La API puede recibir valores de 'metascore', 'year' y 'genres' como entrada para obtener la predicción del precio de un juego.
- La API devuelve la predicción del precio y el RMSE durante el entrenamiento del modelo como respuesta.


En los siguientes enlaces se pueden visualizar los códigos realizados:

Modelo [aquí](https://github.com/Angiea18/1-proyecto-individual-MLOps/blob/main/steam_games_ML.ipynb)

API [aquí](https://github.com/Angiea18/1-proyecto-individual-MLOps/blob/main/main.py)

## Deployment
Para el deploy de la API, se utilizó la plataforma **`Render`**. Los datos están listos para ser consumidos y consultados a partir del siguiente link

[Link al Deployment](https://steam-games.onrender.com/)
## Video
Para consultar sobre los pasos del proceso y una explicación más profunda es posible acceder al [siguiente enlace]()
