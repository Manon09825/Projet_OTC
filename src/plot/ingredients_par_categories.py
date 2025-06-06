import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

""" Ce script calcule le nombre moyen d'ingrédients par catégorie (entrée, plat, dessert). Il sauvegarde les résultats dans le dossier figures sous forme de fichier png. """

df = pd.read_csv("../../data/clean/minimalist_baker_recipes_clean.csv")

df["num_ingredients"] = df["ingredients_clean"].apply(lambda x: len(x.split()))
df.groupby("label")["num_ingredients"].mean().plot(kind="bar", title="Nombre moyen des ingrédients par catégorie")
plt.xlabel('Catégorie')
plt.ylabel("Nombre d'ingrédients")
plt.xticks(rotation=0)
plt.savefig('../../figures/ingredients_par_categories.png')
