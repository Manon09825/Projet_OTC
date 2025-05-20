# Projet_OTC
Projet du cours Outils de traitement de corpus (M1 S2 TAL)

## But du projet
- Le but de mon projet est d’entraîner un classifieur qui trie des recettes et les range dans 3 catégories (entrée, plat, dessert).

## Données et besoins
- Les données que je vais récupérer sont essentiellement des données textuelles, extraites de sites de recettes. J'utiliserai la librairie Scrapy pour être sûre de respecter les robots.txt des sites et de scraper le web de façon éthique. 
- Même si le but du classifieur n’est pas de répondre à un besoin vital, il pourra quand même être utile. Par exemple, beaucoup de sites de recettes n’indiquent pas la catégorie de leur recette. Si ce n’est pas gênant quand on cherche une recette en particulier, cela peut l’être si l’on veut chercher une recette de dessert sans avoir d’idée précise. Cela permettrait d’améliorer les filtres de recherche et les systèmes de recommandation.

## Processus
- Les premières étapes sont le crawling et le scraping du web pour récupérer les données dont j'aurai besoin.
- Une fois les données récupérées, le script de visualisation permet de réaliser des statistiques sur le corpus et de les visualiser.
- Vient ensuite l'étape d'augmentation des données grâce à la création d'un dataset synthétique.
- Le script ?? permet d'adapter un modèle transformer à mes tâches et sur mes données.
- Enfin, on procède à l'évaluation du modèle avec le script evaluation.py.
