import ast
import pandas as pd
import numpy as np
from fastapi import FastAPI 
from fastapi import FastAPI, Query, HTTPException
from fastapi.templating import Jinja2Templates 
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
@app.get('/ Earlyacces')
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
                            "Negative", "Very Negative", "Mostly Negative", "Overwhelmingly Positive"]
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


# Asegurarse de que 'release_date' sea de tipo datetime
df_limpio['release_date'] = pd.to_datetime(df_limpio['release_date'])

def train_bagging_model(genres, metascore, year):
    # Filtrar los registros con valores no nulos en la columna 'genres'
    df_filtrado = df_limpio[df_limpio['genres'].notnull()]

    # Paso 1: Filtrar el DataFrame según el género y la disponibilidad anticipada
    df_filtrado = df_filtrado[(df_filtrado['genres'].apply(lambda x: genres in x if isinstance(x, list) else False))]

    # Paso 2: Preparar los datos (asegúrate de haber realizado las transformaciones previas)
    df_filtrado['year'] = df_filtrado['release_date'].dt.year  # Crear una nueva columna 'year' con el año extraído de 'release_date'
    X = df_filtrado[['metascore', 'year']]  # Incluir las variables metascore, year y early_access

    # Obtener todos los géneros únicos presentes en el DataFrame
    all_genres = set()
    for genre_list in df_filtrado['genres']:
        if isinstance(genre_list, list):
            all_genres.update(genre_list)
    all_genres = sorted(list(all_genres))

    # Crear columnas para cada género utilizando one-hot encoding con prefijo 'genre'
    genres_encoded = pd.get_dummies(df_filtrado['genres'].apply(pd.Series).stack(), prefix='genre').groupby(level=0).sum()

    # Combinar las características codificadas con la matriz X
    X = pd.concat([X, genres_encoded], axis=1)

    y = df_filtrado['price']

    # Eliminar filas con valores faltantes
    X.dropna(inplace=True)
    y = y[X.index]

    # Paso 3: Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Paso 4: Crear el modelo base de regresión lineal
    base_model = LinearRegression()

    # Paso 5: Crear el modelo de Bagging utilizando el modelo base
    bagging_model = BaggingRegressor(base_model, n_estimators=10, random_state=42)

    # Paso 6: Entrenar el modelo de Bagging con los datos de entrenamiento
    bagging_model.fit(X_train, y_train)

    # Paso 7: Realizar predicciones con el modelo de Bagging
    y_test_pred = bagging_model.predict(X_test)

    # Paso 8: Calcular el RMSE
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)

    # Guardar el modelo entrenado utilizando pickle
    with open('bagging_model.pkl', 'wb') as model_file:
        pickle.dump(bagging_model, model_file)

    return rmse

# Ejemplo de uso:
genero_elegido = 'Action'
metascore_elegido = 85.0
year_elegido = 2017

rmse_result = train_bagging_model(genero_elegido, metascore_elegido, year_elegido)
print("RMSE del modelo de Bagging:", rmse_result)


# Cargar el modelo entrenado con pickle
with open('bagging_model.pkl', 'rb') as model_file:
    bagging_model = pickle.load(model_file)

# Lista de valores disponibles para el género
available_genres = ['Action', 'Adventure', 'Casual', 'Early Access', 'Free to Play', 'Indie',
                    'Massively Multiplayer', 'RPG', 'Racing', 'Simulation', 'Sports', 'Strategy', 'Video Production']

@app.get("/")
async def read_root():
    return {"message": "¡Bienvenido a la API de predicciones de precios de juegos!"}

@app.get("/predict/")
async def predict_price(genres: str, metascore: float, year: int):
    # Verificar que el valor de género esté dentro de los géneros disponibles
    genres_list = genres.split(",")
    for genre in genres_list:
        if genre not in available_genres:
            raise HTTPException(status_code=422, detail=f"Invalid genre: {genre}. It should be one of the available genres.")

    # Crear la matriz de características para hacer la predicción
    X = [[metascore, year]]
    # Agregar columnas para los géneros, todas con valor 0 (no seleccionados)
    for genre in available_genres:
        X[0].append(1 if genre in genres_list else 0)

    # Realizar la predicción utilizando el modelo de Bagging
    predicted_price = bagging_model.predict(X)[0]

    # Calcular el RMSE para mostrarlo en la respuesta
    rmse_result = 0.0  # Asegurémonos de que sea un valor numérico
    try:
        # Cargar el RMSE del archivo de texto (si existe)
        with open('bagging_rmse.txt', 'r') as rmse_file:
            rmse_result = float(rmse_file.read())
    except FileNotFoundError:
        pass

    return {"predicted_price": predicted_price, "rmse": rmse_result}

