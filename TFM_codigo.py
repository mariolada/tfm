import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.width', None)  # Ajusta el ancho a la consola

# -------------------------------------------------------------------------
# APARTADO 1: MODELO PREDICTIVO
# -------------------------------------------------------------------------
print("------------ 1. Modelo Predictivo ------------")
# Leer el dataset
file_path = "dataset_imputaciones.csv"
df = pd.read_csv(file_path)

# Paso 1: Añadimos la columna `Total_Prevalence` como la suma de las tasas de prevalencia de diferentes trastornos.
prevalence_cols = [
    "Drug.use.disorders",
    "Schizophrenia",
    "Eating.disorders",
    "Bipolar.disorder",
    "Depression",
    "Anxiety.disorders",
    "Alcohol.use.disorders",
]
df["Total_Prevalence"] = df[prevalence_cols].sum(axis=1, skipna=True)

# Paso 2: Codificamos la columna `CountryName` utilizando one-hot encoding para incluir información por país.
df_encoded = pd.get_dummies(df, columns=["CountryName"], drop_first=True)

# Seleccionamos las variables independientes y la dependiente para el modelo.
features = ["EdV", "PIB", "Gini", "Tasa_de_desempleo"] + [col for col in df_encoded.columns if col.startswith("CountryName_")]
features = [col for col in features if col in df_encoded.columns]  # Filtramos las columnas existentes
df_encoded = df_encoded.dropna(subset=features + ["Total_Prevalence"])  # Eliminamos filas con valores faltantes

# Dividimos los datos en conjuntos de entrenamiento y prueba.
X = df_encoded[features]
y = df_encoded["Total_Prevalence"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Probar diferentes modelos de predicción.
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
}

results = {}

# Entrenamos y evaluamos cada modelo.
for name, model in models.items():
    model.fit(X_train, y_train)  # Entrenamos el modelo
    y_pred = model.predict(X_test)  # Hacemos predicciones
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Calculamos el RMSE
    results[name] = rmse
    print(f"Modelo: {name}, Error cuadrático medio (RMSE): {rmse}")

# Guardamos el dataset procesado para pasos posteriores.
df_encoded.to_csv("dataset_procesado_con_paises.csv", index=False)
print("Archivo procesado guardado como `dataset_procesado_con_paises.csv`")


# -------------------------------------------------------------------------
# APARTADO 2: CLÚSTERES Y RELACIONES SOCIOECONÓMICAS EN LA SALUD MENTAL
# -------------------------------------------------------------------------
print("------------ 2. Clústeres y relaciones socioeconómicas ------------")

# Selección de variables relevantes para el análisis de clústeres
# Incluimos variables relacionadas con prevalencias para explorar si afectan los clústeres
clustering_features = ["Total_Prevalence", "Depression", "Anxiety.disorders", "Alcohol.use.disorders", "TD", "PIB"]
clustering_features = [col for col in clustering_features if col in df_encoded.columns]

# Verificar si las columnas necesarias están disponibles
if not clustering_features:
    raise ValueError("No se encontraron las columnas necesarias para el clustering.")

# Combinar columnas de países (CountryName_*) en una única columna 'CountryName'
# Esto nos permite trabajar con una columna más limpia y clara para identificar países
country_columns = [col for col in df_encoded.columns if col.startswith("CountryName_")]
if country_columns:
    # Crear columna 'CountryName' a partir de las columnas dummy
    df_encoded["CountryName"] = df_encoded[country_columns].idxmax(axis=1).str.replace("CountryName_", "")
    # Eliminar las columnas dummy originales
    df_encoded.drop(columns=country_columns, inplace=True)
else:
    raise ValueError("No se encontraron columnas 'CountryName_*' en el DataFrame.")

# Rellenar valores faltantes con la media
# Esto asegura que no haya problemas al normalizar o aplicar K-Means
df_encoded[clustering_features] = df_encoded[clustering_features].fillna(df_encoded[clustering_features].mean())

# Normalizar las variables seleccionadas
# La normalización garantiza que todas las variables tengan la misma escala, evitando sesgos en el clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_encoded[clustering_features])

# Método del codo para determinar el número óptimo de clusters
inertia = []
k_range = range(1, 11)  # Probamos de 1 a 10 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Método del Codo para Determinar el Número de Clusters')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.grid(True)
plt.show()

# Elegir el número óptimo de clusters según el método del codo
n_clusters_optimo = 3  # Ajustar según el gráfico del codo

# Aplicar KMeans con el número óptimo de clusters
df_encoded["Cluster"] = KMeans(n_clusters=n_clusters_optimo, random_state=42).fit_predict(data_scaled)

# Evaluar la calidad de los clusters con el Silhouette Score
silhouette_avg = silhouette_score(data_scaled, df_encoded["Cluster"])
print(f"Silhouette Score: {silhouette_avg}")

# Reducir la dimensionalidad para visualización (PCA)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)
df_encoded["PCA1"] = pca_result[:, 0]
df_encoded["PCA2"] = pca_result[:, 1]

# Visualizar los clusters en el espacio PCA
plt.figure(figsize=(10, 6))
plt.scatter(df_encoded["PCA1"], df_encoded["PCA2"], c=df_encoded["Cluster"], cmap="viridis", alpha=0.7)
plt.title("Clustering de países basado en todas las variables (espacio PCA)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar(label="Cluster")
plt.show()

# ANALIZAMOS LAS PREVALENCIAS EN LOS CLÚSTERES
# Queremos entender si las variables relacionadas con enfermedades influyen en los clústeres
prevalence_vars = ["Depression", "Anxiety.disorders", "Alcohol.use.disorders"]

# Calculamos los promedios de las prevalencias por clúster
cluster_means = df_encoded.groupby("Cluster")[prevalence_vars].mean()
print("\nPromedio de prevalencias por clúster:")
print(cluster_means)

# Graficamos las prevalencias promedio por clúster para identificar patrones
cluster_means.plot(kind="bar", figsize=(10, 6))
plt.title("Promedio de prevalencias por clúster")
plt.xlabel("Clúster")
plt.ylabel("Promedio")
plt.legend(title="Variable")
plt.show()

# GUARDAR RESULTADOS
# Guardamos el DataFrame completo para revisar la asignación de países a clústeres
with open("output.txt", "w") as f:
    f.write(df_encoded.to_string())

# Listar los países por clúster y guardar en un archivo CSV
cluster_countries = df_encoded[["CountryName", "Cluster"]].drop_duplicates()
for cluster in sorted(df_encoded["Cluster"].unique()):
    print(f"\nPaíses en el clúster {cluster}:")
    print(cluster_countries[cluster_countries["Cluster"] == cluster]["CountryName"].values)

cluster_countries.to_csv("countries_by_cluster.csv", index=False)
print("Archivo guardado: countries_by_cluster.csv")



# -------------------------------------------------------------------------
# APARTADO 3: RELACIÓN ENTRE TASA DE DESEMPLEO Y PREVALENCIA TOTAL
# -------------------------------------------------------------------------
print("------------ 3. RELACIÓN ENTRE TASA DE DESEMPLEO Y PREVALENCIA TOTAL ------------")
# Relación entre la Tasa de Desempleo (TD) y la Prevalencia Total de Enfermedades Mentales
if "TD" in df_encoded.columns and "Total_Prevalence" in df_encoded.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_encoded, x="TD", y="Total_Prevalence", hue="Cluster", palette="viridis", alpha=0.7)
    plt.title("Relación entre Tasa de Desempleo (TD) y Prevalencia Total")
    plt.xlabel("Tasa de Desempleo (TD)")
    plt.ylabel("Prevalencia Total de Enfermedades Mentales")
    plt.legend(title="Clúster", loc="best")
    plt.grid()
    plt.show()
else:
    print("Las columnas necesarias ('TD' o 'Total_Prevalence') no están disponibles en el dataset.")






print("--------- GENERACION DE HIPOTESIS Y SIMULACIIONES ---------")
# -------------------------------------------------------------------------
# APARTADO 3: GENERACIÓN DE HIPÓTESIS Y SIMULACIONES
# -------------------------------------------------------------------------


# Selección del modelo base (Random Forest, el más efectivo previamente)
simulation_features = ["EdV", "GDP per capita (current US$)", "gini", "TD"]  # Actualizamos a la nueva columna de GDP

# Verificamos que las columnas necesarias existan en el dataset
simulation_features = [col for col in simulation_features if col in df.columns]

# Dividimos el dataset en variables independientes (X) y dependientes (y)
X = df[simulation_features]
y = df["Total_Prevalence"]

# Entrenamos el modelo de Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

# -------------------------------------------------------------------------
# Simulación 1: Reducir la Tasa de Desempleo (TD) en un 5%
# -------------------------------------------------------------------------

simulation_df = X.copy()
simulation_df["TD"] = simulation_df["TD"] * 0.95

simulation_df["Predicted_Prevalence"] = rf_model.predict(simulation_df)
original_prevalence = y.mean()
simulated_prevalence_td = simulation_df["Predicted_Prevalence"].mean()
change_percentage_td = ((simulated_prevalence_td - original_prevalence) / original_prevalence) * 100

print(f"Cambio promedio en la prevalencia tras reducir el desempleo en un 5%: {change_percentage_td:.2f}%")


# -------------------------------------------------------------------------
# Simulación 2: Aumentar el PIB per cápita en un 10%
# -------------------------------------------------------------------------

if "GDP per capita (current US$)" in X.columns:
    simulation_df_gdp = X.copy()
    simulation_df_gdp["GDP per capita (current US$)"] = simulation_df_gdp["GDP per capita (current US$)"] * 1.10

    simulation_df_gdp["Predicted_Prevalence"] = rf_model.predict(simulation_df_gdp)
    simulated_prevalence_gdp = simulation_df_gdp["Predicted_Prevalence"].mean()
    change_percentage_gdp = ((simulated_prevalence_gdp - original_prevalence) / original_prevalence) * 100

    print(f"Cambio promedio en la prevalencia tras aumentar el PIB per cápita en un 10%: {change_percentage_gdp:.2f}%")
else:
    print("La columna 'GDP per capita (current US$)' no está disponible en el dataset.")
    simulated_prevalence_gdp = None

# -------------------------------------------------------------------------
# Visualización de las simulaciones
# -------------------------------------------------------------------------

scenarios = ["Original", "Reducir TD 5%"]
prevalence_values = [original_prevalence, simulated_prevalence_td]

if simulated_prevalence_gdp is not None:
    scenarios.append("Aumentar PIB 10%")
    prevalence_values.append(simulated_prevalence_gdp)

plt.figure(figsize=(8, 5))
plt.bar(scenarios, prevalence_values, color=["blue", "green", "orange", "purple"])
plt.title("Comparación de Prevalencia Total en Diferentes Escenarios")
plt.ylabel("Prevalencia Total Promedio")
plt.xlabel("Escenario")
plt.ylim(min(prevalence_values) * 0.95, max(prevalence_values) * 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


