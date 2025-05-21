import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

""" Taille des recettes en nombre de mots, par catégorie """

df = pd.read_csv("../../data/clean/minimalist_baker_recipes_clean.csv")

df["text_length"] = (df["ingredients_clean"] + " " + df["instructions_clean"]).apply(lambda x: len(x.split()))
sns.boxplot(x="label", y="text_length", data=df, palette="pastel")
plt.title("Taille moyenne des recettes par catégorie")
plt.xlabel("Catégorie")
plt.ylabel("Nombre de mots")
plt.grid(True)
plt.savefig('../../figures/longueur_recettes.png')
