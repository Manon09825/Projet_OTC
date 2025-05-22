import pandas as pd
import os
import re

""" Ce script de nettoyage fait en sorte de ne garder que des données qui seront utiles à l'entraînement et à l'évaluation des modèles. Le texte est d'abord normalisé, puis le script crée un dataframe pandas qui contient 4 colonnes: le titre de la recette, les ingrédients, les instructions et les labels. Les données sont ensuite sauvegardées dans un fichier csv. """

RAW_PATH = "../../data/raw/minimalist_baker_recipes.csv"
CLEAN_PATH = "../../data/clean/minimalist_baker_recipes_clean.csv"

COURSE_MAP = {
    "appetizer": "entrée",
    "starter": "entrée",
    "side": "entrée",
    "soup": "entrée",
    "snack": "entrée",
    "entree": "plat",
    "main": "plat",
    "dinner": "plat",
    "lunch": "plat",
    "breakfast": "plat",
    "dessert": "dessert",
    "sweet": "dessert"
}

def clean_course_label(course):
    if not isinstance(course, str):
        return None
    course = course.lower()
    for key in COURSE_MAP:
        if key in course:
            return COURSE_MAP[key]
    return None

def clean_text_list(text_list):
    if not isinstance(text_list, list):
        return ""
    joined = " ".join(text_list)

    cleaned = re.sub(r"[^\w\s]", "", joined.lower())
    return cleaned.strip()

def clean_recipes(df):
    df = df.copy()

    df = df.dropna(subset=["ingredients", "instructions", "course"])

    df["ingredients_clean"] = df["ingredients"].apply(clean_text_list)
    df["instructions_clean"] = df["instructions"].apply(clean_text_list)

    df["label"] = df["course"].apply(clean_course_label)

    df = df.dropna(subset=["label"])

    df = df[["title", "ingredients_clean", "instructions_clean", "label"]]

    return df

def main():
    df = pd.read_csv(RAW_PATH, converters={"ingredients": eval, "instructions": eval})
    cleaned_df = clean_recipes(df)


    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    cleaned_df.to_csv(CLEAN_PATH, index=False)
    print(f"Données nettoyées sauvegardées dans{CLEAN_PATH} ({len(cleaned_df)} lignes")

if __name__ == "__main__":
    main()
