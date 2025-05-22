import pandas as pd
from sklearn.model_selection import train_test_split

""" Ce script sépare le corpus nettoyé en deux parties (deux fichiers csv), une pour l'entraînement et une pour le test. Il sépare les informations de chaque recette en deux colonnes, text (les ingrédients, les instructions etc) et label (entrée, plat, dessert), pour que le modèle soit ensuite plus facile à entraîner et évaluer. """

df = pd.read_csv("../../data/clean/minimalist_baker_recipes_balanced.csv")

# Les ingrédients et les instructions contenus dans le fichier d'entrée sont regroupés en une colonne text; la colonne 'label' existe déjà donc pas besoin d'y toucher
df["text"] = df["ingredients_clean"] + " " + df["instructions_clean"]


# On sépare le corpus de départ en deux, on garde 20% pour le test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)


train_df[["text", "label"]].to_csv("../../data/clean/train.csv", index=False)
test_df[["text", "label"]].to_csv("../../data/clean/test.csv", index=False)

print("Les deux datasets ont été créés.")
