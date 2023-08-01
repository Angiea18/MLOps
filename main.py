from fastapi import FastAPI 

app = FastAPI()


@app.get('/ Genero')
def genero(año: str):
    df_filtrado = df[df['release_date'].dt.year == int(año)]
    generos_mas_vendidos = df_filtrado['genres'].explode().value_counts().nlargest(5).index.todict()

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

    print("Los 5 specs más comunes en el año", año, "son:")
    for spec, count in top_5_specs.items():
        print(spec, ":", count)

    return top_5_specs

@app.get('/ Earlyacces')
def earlyacces(año: str):
    df_filtrado = df[(df['release_date'].dt.year == int(año)) & (df['early_access'] == True)]

    cantidad_juegos_early_access = len(df_filtrado)

    return cantidad_juegos_early_access


@app.get('/ Sentimient')
def sentiment(año: str):
    df_filtrado = df[df['release_date'].dt.year == int(año)]

    sentimiento_counts = df_filtrado['sentiment'].value_counts()

    sentimiento_dict = sentimiento_counts.to_dict()

    return sentimiento_dict


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
