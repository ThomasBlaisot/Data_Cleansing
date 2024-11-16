#%% Chargement des données Load
"""
Lire le fichier JSON ou CSV avec Apache Spark. Identifier les colonnes clés
 (e.g., product_name, categories, nutriments, packaging).
"""
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

# Créer une SparkSession
spark = SparkSession.builder \
    .appName("Load CSV with Custom Separator") \
    .getOrCreate()

# Charger le fichier CSV avec un séparateur personnalisé

df_csv = spark.read.format("csv") \
    .option("header", "true") \
    .option("delimiter", "\t") \
    .option("inferSchema", "true") \
    .option("encoding", "UTF-8") \
    .load("en.openfoodfacts.org.products.csv")

# Afficher les 10 premières lignes du DataFrame
df_csv.show(10)
# %%
# Afficher le schéma du DataFrame
df_csv.printSchema()

# %% Nettoyage des données :
"""
Supprimer les lignes avec des valeurs manquantes ou aberrantes. Traiter les doublons sur les produits (product_id ou product_name).
"""
from pyspark.sql.functions import col

columns_to_keep = [
    "product_name",  # Nom du produit (ex. : "Pâtes complètes").
    "categories",  # Catégories auxquelles appartient le produit (ex. : "pâtes", "aliments bio").
    "ingredients_text",  # Liste complète des ingrédients du produit (texte brut).
    "allergens",  # Allergènes présents dans le produit (ex. : "gluten", "arachides").
    "traces",  # Traces potentielles d’allergènes (ex. : "peut contenir des traces de lait").
    "quantity",  # Quantité totale du produit (ex. : "500g", "1L").
    "serving_size",  # Taille d'une portion (ex. : "30g", "1 cuillère à soupe").
    "serving_quantity",  # Quantité par portion (en valeur numérique, ex. : 30 pour 30g).
    "nutriscore_score",  # Score numérique Nutri-Score (plus le score est bas, meilleure est la qualité).
    "nutriscore_grade",  # Grade Nutri-Score (lettre de A à E, où A est le plus sain).
    "nova_group",  # Niveau de transformation du produit (de 1 = peu transformé à 4 = ultra-transformé).
    "ecoscore_score",  # Score numérique Eco-Score, évaluant l’impact environnemental (plus élevé = meilleur impact).
    "ecoscore_grade",  # Grade Eco-Score (lettre de A à E, où A est meilleur pour l’environnement).
    "energy-kj_100g",  # Énergie en kilojoules pour 100g ou 100ml du produit.
    "energy-kcal_100g",  # Énergie en kilocalories pour 100g ou 100ml du produit.
    "sugars_100g",  # Quantité totale de sucres pour 100g ou 100ml.
    "fiber_100g",  # Quantité de fibres alimentaires pour 100g ou 100ml.
    "proteins_100g",  # Quantité totale de protéines pour 100g ou 100ml.
    "salt_100g",  # Quantité totale de sel pour 100g ou 100ml.
    "fat_100g",  # Quantité totale de lipides (matières grasses) pour 100g ou 100ml.
    "saturated-fat_100g",  # Quantité d’acides gras saturés pour 100g ou 100ml.
]

# Filtrer les colonnes
df_filtered = df_csv.select([col for col in columns_to_keep if col in df_csv.columns])
df_filtered = df_filtered.withColumn("quantity", col("quantity").cast("int"))
# Nettoyage des données integer and string
df_clean = df_filtered.na.fill(0, [
    "quantity", "serving_quantity", "energy-kj_100g", "energy-kcal_100g",
    "fat_100g", "saturated-fat_100g", "sugars_100g",
    "fiber_100g", "proteins_100g", "salt_100g",
]) \
    .na.fill("undefined", [
        "product_name", "categories", "ingredients_text", "allergens", "traces",
        "nutriscore_grade", "ecoscore_grade"
    ]) \
    .dropDuplicates()

df_clean.printSchema() # verification de la conversion de type


# Liste des colonnes numériques
numeric_columns = [
    "quantity", "serving_quantity", "nutriscore_score", "ecoscore_score",
    "energy-kj_100g", "energy-kcal_100g", "sugars_100g", "fiber_100g",
    "proteins_100g", "salt_100g", "fat_100g", "saturated-fat_100g"
]

outliers = {}
for column in numeric_columns:
    quantiles = df_clean.approxQuantile(column, [0.25, 0.75], 0.01)  # Précision à 1%
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filtrer les valeurs aberrantes
    outliers[column] = df_clean.filter((col(column) < lower_bound) | (col(column) > upper_bound))

    print(f"Valeurs aberrantes pour la colonne '{column}':")
    outliers[column].show(5)


#%% Normalisation des colonnes :
"""
Extraire des informations pertinentes des colonnes complexes,
 comme categories_tags ou nutriments. Convertir les colonnes au format adéquat 
 (e.g., unités des nutriments en grammes ou millilitres).
"""

from pyspark.sql.functions import col, split, avg

# Extraire la première catégorie
df_clean = df_clean.withColumn("main_category", split(col("categories_tags"), ",").getItem(0))

# %% Liste des colonnes :
"""
Récupérer la liste des noms de colonnes du DataFrame nettoyé.
"""

# Afficher la liste des noms de colonnes
columns_list = df_clean.columns
print("Liste des colonnes :", columns_list)

num_columns = len(columns_list)
print("Nombre total de colonnes :", num_columns)

# Calculer un score nutritionnel basé sur les nutriments
df_clean = df_clean.withColumn("nutrition_score", col("energy_100g") - col("fiber_100g") + col("sugars_100g"))

# Statistiques descriptives par catégorie
stats = df_clean.groupBy("main_category").agg(
    avg("energy_100g").alias("avg_energy"),
    avg("sugars_100g").alias("avg_sugars"),
    avg("fiber_100g").alias("avg_fiber")
)
stats.show()

# %% Transformation des données Transform :
"""Ajouter des colonnes calculées, par exemple : Indice de qualité nutritionnelle 
Calculer un score basé sur les nutriments (e.g., sodium, sugar, fiber). 
Extraire la catégorie principale d'un produit (e.g., "boissons", "snacks"). 
Regrouper les données par catégories (categories) pour analyser les tendances (e.g., moyenne des calories par catégorie).

--> Quel calcules effectuer ?
--> Quel catégories créer ?
"""

# %% Analyse exploratoire :
"""Utiliser des fonctions de calcul sur fenêtre pour : 
Trouver les produits les plus caloriques par catégorie. 
Identifier les tendances de production par brands (marques). 
Générer des statistiques descriptives (e.g., médiane, moyenne des nutriments par catégorie)."""

# %% Sauvegarde des données Save :
"""Partitionner les données par catégories (categories) et années (year). 
Sauvegarder les résultats transformés en format Parquet avec compression Snappy. 
Sauvegarder les résultats transformés dans les bases de données: postgresql/sqlserver/mysql/Snowflake/BigQuery."""

# %% Présentation des résultats :
"""Visualiser les résultats sous forme de graphiques ou tableaux 
(les étudiants peuvent utiliser un outil comme Jupyter Notebook en local ou Google Colab )."""