import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np
import csv


def parser():
    parser = argparse.ArgumentParser(
        description="Statistiques descriptives du corpus")
    parser.add_argument("-p", "--boxplots",
                        action="store_true", help="Affichage des boxplots")
    parser.add_argument("-c", "--csv", type=str, help="Chemin de l'output csv")
    parser.add_argument("input_directory",
                        help="Chemin vers le dossier contenant le corpus") 
    args = parser.parse_args()
    
    return args


def extraction_wc(dossier:Path)->list:
    """extraction des longueurs des textes d'un dossier"""
    
    liste_wc = []
    liste_path_textes = [f for f in dossier.iterdir() if f.is_file()]
    for texte in liste_path_textes:
        with open(texte, "r", encoding = "utf-8") as file:
            fichier_wc = len(file.read().split())
            # taille minimale des textes : valeur déterminée après expérimentation pour garder plus de 500 textes par catégorie
            if fichier_wc >= 16:
                liste_wc.append(fichier_wc)
                
    print(f"{dossier.name.upper()}:")
    print(f"Nombre de textes : {len(liste_wc)}")
    
    return liste_wc


def plot_descriptifs(data: dict):
    """affichage des statistiques descriptives des datas d'un dictionnaire dans
    des boxplots"""
    
    # Création d'une liste géante pour pouvoir plus tard exclure les données
    # abérantes
    all_data = np.concatenate(list(data.values()))
    fig, ax = plt.subplots()
    
    # Création des plots et coloration des boites
    boxplot = ax.boxplot(data.values(), patch_artist=True, showmeans=True)
    
    # Ajustement de la fenêtre pour exclure du visuel les 5% de données supérieures
    # (qui sont abérantes)
    limite_superieure = np.percentile(all_data, 95)
    ax.set_ylim( 0, limite_superieure)
    
    # Changement des couleurs
    for box in boxplot['boxes']:
        box.set(facecolor='cyan')
    for median in boxplot['medians']:
        median.set(color='black')
    for mean in boxplot['means']:
        mean.set(color='white') 
    
    # Assignation des clés du dictionnaire aux labels des plots
    ax.set_xticklabels(data.keys())

    plt.show()
    
    
def write_csv(data:dict, chemin:Path):
    with open(chemin, mode='a', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
    
        writer.writerow(["",
                         "LONGUEUR MOYENNE",
                         "LONGUEUR MAX",
                         "LONGUEUR MIN",
                         "ECART TYPE",
                         "LONGUEUR MEDIANE",
                         "PREMIER QUARTILE",
                         "TROISIEME QUARTILE"])
    
        for categorie in data.keys():
            writer.writerow([
                categorie.upper(),
                np.mean(data[categorie]),
                max(data[categorie]),
                min(data[categorie]),
                np.std(data[categorie]),
                np.mean(data[categorie]),
                np.percentile((data[categorie]), 25),
                np.percentile((data[categorie]), 75),
                ])
            
    return print(f"Le fichier csv a bien été créé au chemin {chemin}")    



def main(dossier=None, plot=None, csv=None):
    dossiers_categories = [d for d in dossier.iterdir() if d.is_dir()]

    #Statistiques vers sortie standard
    print("\nSTATISTIQUES DESCRIPTIVES POUR NOS CATEGORIES : \n")
    data_plot = {}
    for categorie in dossiers_categories:
        if categorie.name != "Adventure":
            categorie_wc_list = extraction_wc(categorie)
            data_plot[categorie.name] = categorie_wc_list
            print(f"longueur moyenne : {np.mean(categorie_wc_list)}")
            print(f"longueur max : {max(categorie_wc_list)}")
            print(f"longueur min : {min(categorie_wc_list)}")
            print(f"écart type : {np.std(categorie_wc_list)}")
            print(f"longueur médiane : {np.percentile(categorie_wc_list, 50)}")
            print(f"premier quartile : {np.percentile(categorie_wc_list, 25)}")
            print(f"troisième quartile : {np.percentile(categorie_wc_list, 75)}\n")

    #Statistiques vers sortie csv
    if csv:
        write_csv(data_plot, Path(csv))
    
    #Statistiques : affichage des plots
    if plot:
        plot_descriptifs(data_plot)


if __name__ == "__main__":
    args=parser()
    print(args)
    main(dossier=Path(args.input_directory), plot=args.boxplots, csv=args.csv)
