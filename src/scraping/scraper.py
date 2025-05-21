import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

BASE_URL = "https://minimalistbaker.com/recipe-index/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Respect des robots.txt
DISALLOWED_PATHS = [
    "/r/", "/ecourse/", "/university/", "/18190176/",
    "/minimalist-baker-detox-guide/", "/m/", "/m/Minimalist-Baker-Detox-Guide.pdf"
]

def is_allowed(url):
    return not any(url.startswith(BASE_URL + path) for path in DISALLOWED_PATHS)

def get_recipe_links():
    print("Recherche des liens à partir de la page d'index...")
    liens = requests.get(BASE_URL, headers=HEADERS)

    soup = BeautifulSoup(liens.text, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("https://minimalistbaker.com/") and not any(x in href for x in ['/m/', '/ecourse/', '/university/']):
            links.append(href)

    # Debugging
    print(f"{len(links)} liens trouvés")
    return links

def scrape_recipe(url):
    """ Fonction qui scrape les pages de recettes """

    liens = requests.get(url, headers=HEADERS)

    soup = BeautifulSoup(liens.text, "html.parser")

    # Extraction du titre
    title_tag = soup.find("h1", class_="entry-title")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown"

    # Extraction des ingrédients
    ingredients = [li.get_text(strip=True) for li in soup.select("ul.wprm-recipe-ingredients li")]

    # Extraction des instructions
    instructions = [li.get_text(strip=True) for li in soup.select("ul.wprm-recipe-instructions li")]

    return {
        "title": title,
        "url": url,
        "ingredients": ingredients,
        "instructions": instructions
    }

def main():
    """Cette fonction scrape et sauvegarde les données scrapées en CSV"""
    links = get_recipe_links()


    recettes = []

    for i, link in enumerate(links):
        try:
            print(f"Scraping {i+1}/{len(links)}: {link}")
            data = scrape_recipe(link)
            recettes.append(data)
            time.sleep(1)
        except Exception as e:
            print(f"Failed to scrape {link}: {e}")

    # Sauvegarde des données en CSV
    output_path = "../../data/raw/minimalist_baker_recipes.csv"
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


    df = pd.DataFrame(recettes)
    df.to_csv(output_path, index=False)
    print(f"{len(df)} recettes sauvegardées dans {output_path}")

if __name__ == "__main__":
    main()

