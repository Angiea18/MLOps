from fastapi import FastAPI 

app = FastAPI()


@app.get('/genero/ Genero')
def genero(año: str):
    df_filtrado = df[df['release_date'].dt.year == int(año)]
    generos_mas_vendidos = df_filtrado['genres'].explode().value_counts().nlargest(5).index.todict()

    return generos_mas_vendidos

@app.get('/juegos/ Juegos')
def juegos(año: str):
    df_filtrado = df[df['release_date'].dt.year == int(año)]
    juegos_lanzados = df_filtrado['title'].tolist()

    return juegos_lanzados


@app.get('/specs/ Specs')
def specs(año: str):
    df_filtrado = df[df['release_date'].dt.year == int(año)]
    specs_list = df_filtrado['specs'].explode().tolist()

    specs_count = pd.Series(specs_list).value_counts()

    specs_mas_comunes = specs_count.nlargest(5).index.todict()

    return specs_mas_comunes

@app.get('/earlyacces/ Earlyacces')
def earlyacces(año: str):
    df_filtrado = df[(df['release_date'].dt.year == int(año)) & (df['early_access'] == True)]

    cantidad_juegos_early_access = len(df_filtrado)

    return cantidad_juegos_early_access


@app.get('/sentiment/ Sentimient')
def sentiment(año: str):
    df_filtrado = df[df['release_date'].dt.year == int(año)]

    sentimiento_counts = df_filtrado['sentiment'].value_counts()

    sentimiento_dict = sentimiento_counts.to_dict()

    return sentimiento_dict


@app.get('/metascore/ Metascore')
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