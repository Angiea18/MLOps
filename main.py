from fastapi import FastAPI 

app = FastAPI()


@app.get('/genero/{ano}')
def genero(ano: str):
    df_filtrado = df[df['release_date'].dt.year == int(ano)]
    generos_mas_vendidos = df_filtrado['genres'].explode().value_counts().nlargest(5).index.tolist()

    return generos_mas_vendidos

@app.get('/juegos/{ano}')
def juegos(ano: str):
    df_filtrado = df[df['release_date'].dt.year == int(ano)]
    juegos_lanzados = df_filtrado['title'].tolist()

    return juegos_lanzados


@app.get('/specs/{ano}')
def specs(ano: str):
    df_filtrado = df[df['release_date'].dt.year == int(ano)]
    specs_list = df_filtrado['specs'].explode().tolist()

    specs_count = pd.Series(specs_list).value_counts()

    specs_mas_comunes = specs_count.nlargest(5).index.todict()

    return specs_mas_comunes

@app.get('/earlyacces/{ano}')
def earlyacces(ano: str):
    df_filtrado = df[(df['release_date'].dt.year == int(ano)) & (df['early_access'] == True)]

    cantidad_juegos_early_access = len(df_filtrado)

    return cantidad_juegos_early_access


@app.get('/sentiment/{ano}')
def sentiment(ano: str):
    df_filtrado = df[df['release_date'].dt.year == int(ano)]

    sentimiento_counts = df_filtrado['sentiment'].value_counts()

    sentimiento_dict = sentimiento_counts.to_dict()

    return sentimiento_dict


@app.get('/metascore/{ano}')
def metascore(ano: str):
    df_filtrado = df[df['release_date'].dt.year == int(ano)].sort_values(by='metascore', ascending=False)
    top_juegos_metascore = df_filtrado.head(5)[['title', 'metascore']].to_dict(orient='records')

    return top_juegos_metascore

