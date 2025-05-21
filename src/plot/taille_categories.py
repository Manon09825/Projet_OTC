import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""Taille des catégories en nombre de recettes"""

df = pd.read_csv("../../data/clean/minimalist_baker_recipes_clean.csv")

df["label"].value_counts().plot(kind="bar", color="salmon", edgecolor="black")
plt.title("Nombre de recettes par catégorie")
plt.xlabel("Catégorie")
plt.ylabel("Recettes")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.savefig('../../figures/recettes_par_categories.png')
