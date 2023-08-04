import ast
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor
from pydantic import BaseModel
import pickle

# Indicamos título y descripción de la API
app = FastAPI(title='PI N°1 (MLOps) -Angie Arango Zapata DTS13')

#Dataset
games = []
with open('steam_games.json') as f: 
    for line in f.readlines():
        games.append(ast.literal_eval(line)) 
df = pd.DataFrame(games) 
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')


# Endpoint 1: Géneros más repetidos por año
@app.get('/ Genero')
def genero(año: str):
    # Filtrar el DataFrame por el año proporcionado
    df_filtrado = df[df['release_date'].dt.year == int(año)]
    
    # Obtener una lista con todos los géneros para el año dado
    generos_list = df_filtrado['genres'].explode().tolist()

    # Calcular el conteo de cada género en la lista
    generos_count = pd.Series(generos_list).value_counts()

    # Obtener los 5 géneros más vendidos en orden correspondiente
    generos_mas_repetidos = generos_count.nlargest(5).to_dict()

    return generos_mas_repetidos

# Endpoint 2: Juegos lanzados en un año
@app.get('/ Juegos')
def juegos(año: str):
    # Filtrar los datos por el año especificado
    df_filtrado = df[df['release_date'].dt.year == int(año)]

    # Obtiener la lista de juegos lanzados en el año
    juegos_lanzados = df_filtrado['title'].tolist()

    return juegos_lanzados

# Endpoint 3: Specs más comunes por año
@app.get('/ Specs')
def specs(año: str):
    # Filtrar el DataFrame por el año proporcionado
    df_filtrado = df[df['release_date'].dt.year == int(año)]
    
    # Obtener una lista con todos los Specs para el año dado
    specs_list = df_filtrado['specs'].explode().tolist()

    # Calcular el conteo de cada Spec en la lista
    specs_count = pd.Series(specs_list).value_counts()

    # Obtener los 5 Specs más comunes en orden correspondiente
    top_5_specs = specs_count.nlargest(5).to_dict()

    return top_5_specs

# Endpoint 4: Cantidad de juegos con early access en un año
@app.get('/ Earlyaccess')
def earlyacces(año: str):
    #Filtrar los datos por el año especificado y por juegos con early access
    df_filtrado = df[(df['release_date'].dt.year == int(año)) & (df['early_access'] == True)]

    #Contar la cantidad de juegos con early access en el año especificado
    cantidad_juegos_early_access = len(df_filtrado)

    return cantidad_juegos_early_access

# Endpoint 5: Análisis de sentimiento por año
@app.get('/ Sentimient')
def sentiment(año: str):
    # Filtrar los datos por el año especificado
    df_filtrado = df[df['release_date'].dt.year == int(año)]

    # Contar la cantidad de registros que cumplen con cada análisis de sentimiento
    sentimient_counts = df_filtrado['sentiment'].value_counts()

    # Convertir la serie de conteo en un diccionario
    sentimient_dict = sentimient_counts.to_dict()

    # Eliminar sentimientos que no están en la lista mencionada
    sentimient_valid = ["Mixed", "Positive", "Very Positive", "Mostly Positive",
                            "Negative", "Very Negative", "Mostly Negative", "Overwhelmingly Positive", "Overwhelmingly Negative"]
    sentimient_dict = {sentimient: count for sentimient, count in sentimient_dict.items() if sentimient in sentimient_valid}

     # Verificar si el diccionario está vacío
    if not sentimient_dict:
        return "Sin registros"

    return sentimient_dict

# Endpoint 6: Top 5 juegos con mayor metascore por año
@app.get('/ Metascore')
def metascore(año: str):
    #Filtrar los datos por el año especificado y ordena por metascore de forma descendente
    df_filtrado = df[df['release_date'].dt.year == int(año)].sort_values(by='metascore', ascending=False)

    #Seleccionar los 5 juegos con mayor metascore y obtiene su información
    top_juegos_metascore = df_filtrado.head(5)[['title', 'metascore']].to_dict(orient='records')

    return top_juegos_metascore

# Cargar el modelo de regresión lineal múltiple desde el archivo con pickle
modelo_guardado = "modelo_regresion_lineal.pkl"
with open(modelo_guardado, "rb") as file:
    linear_model = pickle.load(file)

# Función para obtener las variables dummy de los géneros
def one_hot_encode(genres):
    # Convertir la entrada a una Serie
    genres = pd.Series(genres)

    # Obtener las variables dummy
    genres_dummies = genres.str.get_dummies(sep=",")

    return genres_dummies

# Función para realizar la predicción
def predict_price(year, metascore, genres):
    # Convertir la entrada a un DataFrame
    data = pd.DataFrame([[year, metascore, genres]], columns=["year", "metascore", "genres"])

    # Obtener las variables dummy de los géneros
    data_genres = one_hot_encode(data["genres"])

    # Verificar si hay géneros adicionales y agregarlos con valor 0
    genres_columns = [col for col in X_train.columns if col.startswith("genres_")]
    for col in genres_columns:
        if col not in data_genres.columns:
            data_genres[col] = 0

    # Concatenar las variables dummy con el resto de las columnas
    data = pd.concat([data[["year", "metascore"]], data_genres], axis=1)

    # Reorganizar las columnas para que coincidan con el orden del entrenamiento
    data = data[X_train.columns]

    # Realizar la predicción
    predicted_price = linear_model.predict(data)[0]

    return predicted_price

# Decorador para validar los tipos de datos de la entrada y la salida
def validate_types(func):
    def wrapper(*args, **kwargs):
        # Validar la entrada
        year, metascore, genres = args
        assert isinstance(year, int), "El año debe ser un entero"
        assert isinstance(metascore, float), "El metascore debe ser un decimal"
        assert isinstance(genres, str), "Los géneros deben ser una cadena"

        # Ejecutar la función original
        result = func(*args, **kwargs)

        # Validar la salida
        assert isinstance(result, float), "El precio predicho debe ser un decimal"

        return result

    return wrapper

# Ruta para la predicción
@app.get("/predict/")
@validate_types # Aplicar el decorador a la función predict
def predict(year: int, metascore: float, genres: str):
    # Obtener la predicción
    predicted_price = predict_price(year, metascore, genres)

    # Calcular el RMSE si tienes las etiquetas verdaderas
    if y_test is not None:
        data_genres = one_hot_encode(genres)
        data = pd.concat([pd.DataFrame([[year, metascore]], columns=["year", "metascore"]), data_genres], axis=1)
        y_pred = linear_model.predict(data)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    else:
        rmse = None

    # Formatear el resultado de la predicción con f-strings
    result = f"El precio predicho es {predicted_price:.2f} dólares. El RMSE es {rmse:.2f}."

    return result
