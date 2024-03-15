import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
import csv


def extraction_wc(dossier:Path)->list:
    liste_wc = []
    liste_path_textes = [f for f in dossier.iterdir() if f.is_file()]
    for texte in liste_path_textes:
        with open(texte, "r", encoding = "utf-8") as file:
            fichier_wc = len(file.read().split())
            liste_wc.append(fichier_wc)
    return liste_wc

def plot_descriptifs(data: dict):
    all_data = np.concatenate(list(data.values()))
    fig, ax = plt.subplots()
    boxplot = ax.boxplot(data.values(), patch_artist=True, showmeans=True)
    limite_superieure = np.percentile(all_data, 95)
    ax.set_ylim(ax.get_ylim()[0], limite_superieure)
    for box in boxplot['boxes']:
        box.set(facecolor='cyan')
    for median in boxplot['medians']:
        median.set(color='black')
    for mean in boxplot['means']:
        mean.set(color='white') 
    ax.set_xticklabels(data.keys())

    plt.show()
    


chemin_dossier = Path(sys.argv[1])
chemin_output = sys.argv[2]

dossiers_categories = [d for d in chemin_dossier.iterdir() if d.is_dir()]
#liste_fichiers = list(chemin_dossier.glob('**/*'))

print("STATISTIQUES DESCRIPTIVES POUR NOS CATEGORIES : \n\n")
data_plot = {}
for categorie in chemin_dossier.iterdir():
    categorie_wc_list = extraction_wc(categorie)
    data_plot[categorie.name] = categorie_wc_list
    print(f"{categorie.name.upper()}\nlongueur moyenne : {np.mean(categorie_wc_list)}\nlongueur max : {max(categorie_wc_list)}\nlongueur min : {min(categorie_wc_list)}\nécart type : {np.std(categorie_wc_list)}\nlongueur médiane : {np.percentile(categorie_wc_list, 50)}\npremier quartile : {np.percentile(categorie_wc_list, 25)}\ntroisième quartile : {np.percentile(categorie_wc_list, 75)}\n")


with open(chemin_output, mode='a', newline='') as fichier_csv:
    writer = csv.writer(fichier_csv)
    
    writer.writerow(["", "LONGUEUR MOYENNE", "LONGUEUR MAX", "LONGUEUR MIN", "ECART TYPE", "LONGUEUR MEDIANE", "PREMIER QUARTILE", "TROISIEME QUARTILE"])
    
    for categorie in data_plot.keys():
        writer.writerow([
            categorie.upper(),
            np.mean(data_plot[categorie]),
            max(data_plot[categorie]),
            min(data_plot[categorie]),
            np.std(data_plot[categorie]),
            np.mean(data_plot[categorie]),
            np.percentile((data_plot[categorie]), 25),
            np.percentile((data_plot[categorie]), 75),
            ])

plot_descriptifs(data_plot)


        


    