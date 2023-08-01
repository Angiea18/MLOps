from fastapi import FastAPI 

app = FastAPI(title='PROYECTO INDIVIDUAL Nº1 -Machine Learning Operations (MLOps) -Angie Arango Zapata DTS13',
            description='API de datos de videojuegos')


@app.get('/ Genero')
def genero(año: str):
    df_filtrado = df[df['release_date'].dt.year == int(año)]
    generos_list = df_filtrado['genres'].explode().tolist()
    generos_count = pd.Series(generos_list).value_counts()
    generos_mas_vendidos = generos_count.nlargest(5).to_dict()

    return generos_mas_vendidos

@app.get('/ Juegos')
def juegos(año: str):
    df_filtrado = df[df['release_date'].dt.year == int(año)]
    juegos_lanzados = df_filtrado['title'].tolist()

    return juegos_lanzados

@app.get('/ Specs')
def specs(año: str):
    df_filtrado = df[df['release_date'].dt.year == int(año)]
    specs_list = df_filtrado['specs'].explode().tolist()
    specs_count = pd.Series(specs_list).value_counts()
    
    top_5_specs = specs_count.nlargest(5).to_dict()
    
    return top_5_specs

@app.get('/ Earlyacces')
def earlyacces(año: str):
    df_filtrado = df[(df['release_date'].dt.year == int(año)) & (df['early_access'] == True)]

    cantidad_juegos_early_access = len(df_filtrado)

    return cantidad_juegos_early_access

@app.get('/ Sentimient')
def sentiment(año: str):
    df_filtrado = df[df['release_date'].dt.year == int(año)]
    sentimient_counts = df_filtrado['sentiment'].value_counts()
    sentimient_dict = sentimient_counts.to_dict()
    sentimient_valid = ["Mixed", "Positive", "Very Positive", "Mostly Positive",
                            "Negative", "Very Negative", "Mostly Negative", "Overwhelmingly Positive"]
    sentimient_dict = {sentimient: count for sentimient, count in sentimient_dict.items() if sentimient in sentimient_valid}

    if not sentimient_dict:
        return "Sin registros"

    return sentimient_dict

@app.get('/ Metascore')
def metascore(año: str):
    df_filtrado = df[df['release_date'].dt.year == int(año)].sort_values(by='metascore', ascending=False)
    top_juegos_metascore = df_filtrado.head(5)[['title', 'metascore']].to_dict(orient='records')

    return top_juegos_metascore


import ast
import pandas as pd

games = []
with open('steam_games.json') as f: 
    for line in f.readlines():
        games.append(ast.literal_eval(line)) 
df = pd.DataFrame(games) 
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
