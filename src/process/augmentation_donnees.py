import pandas as pd
from collections import Counter
import random
from nltk.corpus import wordnet
import random

""" Ce script prend en compte le déséquilibre entre les classes et augmente donc en priorité la classe "plat", qui est sous-représentée dans le corpus d'origine.
Le dataset synthétique est créé en remplaçant des mots du corpus d'origine par des synonymes. """

df = pd.read_csv("../../data/clean/minimalist_baker_recipes_clean.csv")

def synonym_replace(text, n=2):
    words = text.split()
    new_words = words.copy()
    random.shuffle(new_words)
    replaced = 0

    for i, word in enumerate(new_words):
        syns = wordnet.synsets(word)
        lemmas = set(l.name().replace('_', ' ') for s in syns for l in s.lemmas())
        lemmas.discard(word)
        if lemmas:
            new_word = random.choice(list(lemmas))
            index = words.index(word)
            words[index] = new_word
            replaced += 1
        if replaced >= n:
            break
    return " ".join(words)

# On compte le nombre de documents par classe pour pouvoir augmenter les données en conséquence
class_counts = df["label"].value_counts()
max_count = class_counts.max()

augmented_rows = []

for label in df["label"].unique():
    df_class = df[df["label"] == label]
    count = len(df_class)

    num_to_augment = max_count - count

    for _ in range(num_to_augment):
        original = df_class.sample(1).iloc[0].copy()

        original["ingredients_clean"] = synonym_replace(original["ingredients_clean"], n=2)
        original["instructions_clean"] = synonym_replace(original["instructions_clean"], n=2)
        original["title"] += " (augmented)"

        augmented_rows.append(original)

augmented_df = pd.DataFrame(augmented_rows)

# On combine le nouveau dataset avec le dataset d'origine
full_df = pd.concat([df, augmented_df], ignore_index=True)


full_df.to_csv("../../data/clean/minimalist_baker_recipes_balanced.csv", index=False)
print(f"Données originales: {len(df)} — Données augmentées: {len(augmented_df)} — Total: {len(full_df)}")
