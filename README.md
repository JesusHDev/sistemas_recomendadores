Proyecto 6: Sistemas de Recomendación Inteligentes 
Este repositorio implementa un motor de recomendación híbrido que evoluciona desde el análisis estadístico simple hasta el filtrado basado en contenido con Ingeniería de Características (Feature Engineering) avanzada.

 Arquitectura del Sistema
El proyecto se divide en tres niveles de profundidad arquitectónica:

1. Recomendador Estadístico (Simple Recommendation)
Utiliza una Media Ponderada (Weighted Rating) para calcular el éxito real de una película, evitando sesgos por bajo volumen de votos. Se basa en la fórmula original de IMDB:

$$W = \left( \frac{v}{v+m} \cdot R \right) + \left( \frac{m}{v+m} \cdot C \right)$$v: 

Número de votos.m: Umbral mínimo de votos (Cuantil 0.90).R: Calificación promedio de la película.C: Promedio general de votos en todo el dataset.

2. Recomendador Semántico (NLP Overview)
Aplica procesamiento de lenguaje natural sobre las sinopsis para encontrar similitudes latentes:

Vectorización TF-IDF: Convierte texto en una matriz de importancia estadística.

Similitud de Coseno: Mide el ángulo entre vectores en un espacio multidimensional para determinar qué tan cerca están dos tramas.

3. Motor de Metadatos (Metadata Soup)
Nivel avanzado de personalización que fusiona múltiples fuentes de datos (Director, Top 3 Actores, Keywords y Géneros) para crear un "ADN" único por película.

Tokenización: Limpieza de espacios para evitar colisiones de nombres (ej. johnnydepp vs johnnygalecki).

CountVectorizer: Utilizado en lugar de TF-IDF para dar peso directo a la frecuencia de los creadores y géneros favoritos.

Stack Tecnológico
Lenguaje: Python 3.x

Procesamiento de Datos: Pandas, NumPy.

Machine Learning: Scikit-Learn (TfidfVectorizer, CountVectorizer, Cosine Similarity).

Análisis: Jupyter Notebooks / Python Scripts.

Instalación y Uso
Entorno Virtual:

python -m venv venv_sr
source venv_sr/bin/activate

Dependencias:

pip install -r requirements.txt

Ejecución:
Carga el dataset movies_metadata.csv en la carpeta /data y ejecuta el script principal o el notebook.

Resultados Obtenidos
Al procesar "The Dark Knight Rises", el sistema es capaz de recomendar no solo películas de la misma franquicia, sino obras que comparten el estilo de dirección de Christopher Nolan y temáticas de Crime/Action, validando la precisión de la "Metadata Soup".
