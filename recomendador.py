import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- PASO 1: CARGA Y LIMPIEZA DE DATOS ---
def cargar_datos():
    # low_memory=False evita advertencias sobre tipos de datos mixtos
    df = pd.read_csv('data/movies_metadata.csv', low_memory=False)
    
    # Limpiamos filas con datos basura en el ID (necesario en este dataset)
    df = df.drop([19730, 29503, 35587])
    df['id'] = pd.to_numeric(df['id'])
    
    # Llenamos nulos en las descripciones para que el NLP no falle
    df['overview'] = df['overview'].fillna('')
    return df

# --- PASO 2: EL RECOMENDADOR SIMPLE (IMDB WEIGHTED RATING) ---
def recomendador_simple(df):
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.90) # Solo el top 10% de películas con más votos
    
    # Filtramos películas que cumplen con el mínimo de votos
    q_movies = df.copy().loc[df['vote_count'] >= m]
    
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Fórmula de IMDB: Regularización Bayesiana
        return (v/(v+m) * R) + (m/(m+v) * C)

    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
    return q_movies.sort_values('score', ascending=False)

# --- PASO 3: EL RECOMENDADOR SEMÁNTICO (NLP) ---
def motor_similitud_contenido(df):
    # Limitamos a 20,000 para no saturar la RAM si tu equipo es modesto
    df_small = df.head(20000).copy()
    
    # TF-IDF: Convierte texto en una matriz de importancia de palabras
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_small['overview'])
    
    # Calculamos la Similitud de Coseno (es el producto punto de los vectores)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Mapeo de títulos a índices para búsqueda rápida
    indices = pd.Series(df_small.index, index=df_small['title']).drop_duplicates()
    
    return df_small, cosine_sim, indices

def obtener_recomendaciones(titulo, df, cosine_sim, indices):
    if titulo not in indices:
        return "Película no encontrada en el dataset reducido."
    
    idx = indices[titulo]
    # Obtenemos las puntuaciones de similitud de esa película con todas las demás
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordenamos por similitud (el segundo elemento de la tupla)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Tomamos las 10 más similares (excluyendo la primera que es ella misma)
    sim_indices = [i[0] for i in sim_scores[1:11]]
    return df['title'].iloc[sim_indices]

# --- EJECUCIÓN ---
if __name__ == "__main__":
    movies = cargar_datos()
    
    print("Top 5 Películas por Ranking IMDB:")
    print(recomendador_simple(movies)[['title', 'score']].head(5))
    
    print("\nCalculando matriz de similitud semántica...")
    df_red, similitud, ind = motor_similitud_contenido(movies)
    
    peli = "The Dark Knight Rises"
    print(f"\nSi te gustó '{peli}', te recomendamos:")
    print(obtener_recomendaciones(peli, df_red, similitud, ind))
