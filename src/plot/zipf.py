import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

""" Ce script calcule la loi de Zipf (les mots les plus fréquents) par catégorie de recette (entrée, plat ou dessert).
Les résultats sont sauvegardés dans le dossier figures sous forme de fichiers png (un fichier par classe, pour pouvoir visualiser chaque mot avec sa fréquence).
Le script ne prend pas en compte les mots grammaticaux ("and", "or", "the", etc.), peu représentatifs de chaque classe et donc peu pertinents pour la loi de Zipf. """

df = pd.read_csv("../../data/clean/minimalist_baker_recipes_clean.csv")

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def plot_top_words_by_label(df, label, n=15):
    texts = df[df["label"] == label]["ingredients_clean"] + " " + df[df["label"] == label]["instructions_clean"]
    all_words = " ".join(texts).split()

    # On enlève les mots grammaticaux pour ne garder que les mots lexicaux
    words_filtered = [word for word in all_words if word not in stop_words]

    common_words = Counter(words_filtered).most_common(n)

    words, counts = zip(*common_words)
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title(f"Mots les plus fréquents dans la catégorie '{label}'")
    plt.xlabel("Fréquence")
    plt.ylabel("Mot")
    plt.tight_layout()



for label in df["label"].unique():
    plot_top_words_by_label(df, label, n=15)
    plt.savefig(f'../../figures/zipf_{label}.png')
