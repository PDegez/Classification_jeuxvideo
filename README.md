# FOUILLE DE TEXTE : GUIDE D'UTILISATION DES SCRIPTS
Degez Pauline & Fleith Valentine
<br>

__________________________________
## PREAMBULE : ORGANISATION DU DEPOT :

- aux
- src
    - algorithms
    - preprocessing
    - visualization
- plots
    - dimensionality reductions
    - algorithms
    - boxplots_stats
- corpus
    - Comedy
    - Family
    - Fantasy
    - Sci-Fi
- classification_reports

__________________________________
## CREATION ET PRETRAITEMENTS :
Attention : présuppose que vous disposiez de l'arborescence fichier du git en local.
Ces scripts sont dans le dossier src/preprocessing

**Extraction du corpus depuis le fichier csv imdb** :

```bash
python3 make_corpus.py
```

**Nettoyage des données et prétraitements** :

```bash
python3 clean_corpus.py
```

**Elagage du corpus (suppression des textes de moins de 16 mots)** :

```bash
python3 skim_corpus.py ../../corpus
```
______________________________________
## STATISTIQUES ET EXPLORATION DES DONNEES :

Attention : présuppose que vous disposiez de l'arborescence fichier du git en local.
Ces scripts sont dans le dossier src/visualization

**LLE avec 2 components** :

```bash
python3 LLE_2components.py ../../corpus/
```

**PCA 2 components** :

```bash
python3 PCA_2components.py ../../corpus/
```

**PCA_explained_variance** :

```bash
python3 PCA_explained_variance.py ../..corpus/
```

**Statistiques descriptives**

<u>Sortie standard<u> :

```bash
python3 stats_descriptives.py /corpus
```

<u>Sortie csv<u> :

```bash
python3 stats_descriptives.py /corpus -c fichier.csv
```

<u>Affichage des boxplots<u> :

```bash
python3 stats_descriptives.py /corpus -p
```
_______________________________________________________
## CLASSIFIEURS :

Attention : présuppose que vous disposiez de l'arborescence fichier du git en local.
Ces scripts sont dans le dossier src/algorithms.

** Calcul du best k pour knn** :

```bash
python3 python3 best_k.py ../../corpus/
```

** Classifieurs ** :

Ces scripts sont tous lancé à partir d'un master script (main_script.py) qui va appeler des fonctions stockées dans le fichier algorithms.py. Ils affichent tous par défaut la matrice de confusion du classifieur et renvoient en sortie standard les différents scores (precision, recall, accuracy, f1_score etc...).

```bash
main_script.py [-h] [-c {KNN,NB,SVM,DT,RF}] [-o OUTPUT_FILE] input_directory
```

- L'option -c permet de choisir le classifieur :
    - KNN : K voisins
    - NB : Naive Bayes
    - SVM : SVM
    - DT : Decision Tree
    - RF : Random Forest
- l'option -o de générer un rapport avec les différents scores
- l'argument renvoie au corpus

Par exemple, le script suivant va lancer le classifieur Naive Bayes, afficher sa matrice de confusion et enregistrer ses scores dans le fichier score.txt :

```bash
python3 main_script -c NB -o score.txt ../../corpus
```

Le script DecisionTree.py a été conservé afin de garder une trace des hyperparamètres explorés.


