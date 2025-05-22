# Projet_OTC
Projet du cours Outils de traitement de corpus (M1 S2 TAL)

## But du projet
- Le but de mon projet est d’entraîner un classifieur qui trie des recettes et les range dans 3 catégories (entrée, plat, dessert).

## Données et besoins
- Les données utilisées sont des données textuelles, tirées du site [minimalistbaker.com](minimalistbaker.com).
- Les données que je vais récupérer sont des données . J'utiliserai la librairie Scrapy pour être sûre de respecter les robots.txt des sites et de scraper le web de façon éthique. 
- Même si le but du classifieur n’est pas de répondre à un besoin vital, il pourra quand même être utile. Par exemple, beaucoup de sites de recettes n’indiquent pas la catégorie de leur recette. Si ce n’est pas gênant quand on cherche une recette en particulier, cela peut l’être si l’on veut chercher une recette de dessert sans avoir d’idée précise. Cela permettrait d’améliorer les filtres de recherche et les systèmes de recommandation.

## Processus
### Crawling et Scraping
- J'ai commencé par vérifier les robots.txt du site à partir duquel je voulais constituer mon corpus, et j'ai fait en sorte de les respecter et de scraper de façon éthique.
- Le script [`scraper.py`](https://github.com/Manon09825/Projet_OTC/blob/main/src/scraping/scraper.py) crawle et scrape le site minimalistbaker.com en utilisant BeautifulSoup4 et en respectant les robots.txt.

### Nettoyage et division du corpus
- Le script [`cleaner.py`](https://github.com/Manon09825/Projet_OTC/blob/main/src/process/cleaner.py) nettoie le corpus (normalisation, nettoyage des balises html...).
- Le script [`split_corpus.py`](https://github.com/Manon09825/Projet_OTC/blob/main/src/process/split_corpus.py) sépare le corpus en deux dataset: un dataset d'entraînement (80% du corpus d'origine) et un dataset de test (20% du corpus d'origine).

### Statistiques et visualisations
- Une fois les données récupérées, j'ai fait [plusieurs scripts](https://github.com/Manon09825/Projet_OTC/tree/main/src/plot) qui réalisent des statistiques sur le corpus et qui appliquent des visualisations.
- Le script [`zipf.py`](https://github.com/Manon09825/Projet_OTC/blob/main/src/plot/zipf.py) calcule la loi de Zipf pour chaque catégorie de recette.
- [`taille_recettes.py`](https://github.com/Manon09825/Projet_OTC/blob/main/src/plot/taille_recettes.py) calcule la taille moyenne des recettes par catégorie, en nombre de mots.
- [`ingredients_par_categories`](https://github.com/Manon09825/Projet_OTC/blob/main/src/plot/ingredients_par_categories.py) calcule le nombre moyen d'ingrédients par catégorie.
- [`taille_categories.py`](https://github.com/Manon09825/Projet_OTC/blob/main/src/plot/taille_categories.py) calcule la taille des catégories en nombre de recettes, ce qui permet de voir si les classes sont équilibrées et, en l'occurrence, de se rendre compte qu'elle ne le sont pas.
- Tous ces scripts sauvegardent leurs statistiques dans des [fichiers png](https://github.com/Manon09825/Projet_OTC/tree/main/figures) pour permettre la visualisation.

### Augmentation des données
- Après avoir calculé le nombre de documents par classe, on se rend compte que les classes sont déséquilibrées; la classe "plat", notamment, est largement sous-représentée. On va donc se concentrer sur ce problème lors de l'augmentation des données.
- Le script [`augmentation_donnees.py`](https://github.com/Manon09825/Projet_OTC/blob/main/src/process/augmentation_donnees.py) crée un dataset synthétique en faisant une copie du corpus existant et en remplacant les mots par des synonymes (synonym replacement). Il fait en sorte d'augmenter principalement la classe "plat", pour égaliser les proportions.

### Finetuning d'un transformer de HuggingFace
- Le script [`finetuning_transformers.py`](https://github.com/Manon09825/Projet_OTC/blob/main/src/process/finetuning_transformers.py) charge le transformer pré-entraîné DistilBert de HuggingFace. Il l'entraîne à la tâche de classification de recettes sur le dataset d'entraînement du corpus, grâce au Trainer de HuggingFace.
- Le modèle entraîné est sauvegardé dans un dossier bin/my_distilbert_model, qui était trop volumineux pour être chargé sur github.

### Évaluation
- Une fois le modèle entraîné sur le training dataset, il est temps de le tester sur le dataset de test et d'évaluer ses performances.
- Le script [`evaluation.py](https://github.com/Manon09825/Projet_OTC/blob/main/src/process/evaluation.py) commence par tester le modèle sur l'ensemble de test. Puis il calcule la précision, le rappel et la f-mesure, et construit la matrice de confusion.
- Les résultats de l'évaluation sont disponibles sous forme de fichiers png dans le dossier figures ([metrics_summary.png](https://github.com/Manon09825/Projet_OTC/blob/main/figures/metrics_summary.png) pour la précision, le rappel et la f-mesure, [evaluation.png](https://github.com/Manon09825/Projet_OTC/blob/main/figures/evaluation.png) pour la matrice de confusion)
