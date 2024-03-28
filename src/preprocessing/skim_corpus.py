#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:27:29 2024

@author: pauline
"""
import argparse
import os
from pathlib import Path
import shutil



def parser():
    parser = argparse.ArgumentParser(
        description="Corpus V2 : suppression aventure, textes courts")
    parser.add_argument("input_dir", type=str, help="dossier corpus source") 
    args = parser.parse_args()
    
    return args


def nettoyage_corpus(dossier:Path)->list:
    """suppression des fichiers trop courts"""
    liste_path_textes = [f for f in dossier.iterdir() if f.is_file()]
    for texte in liste_path_textes:
        with open(texte, "r", encoding = "utf-8") as file:
            fichier_wc = len(file.read().split())
            # taille minimale des textes : valeur déterminée après expérimentation pour garder plus de 500 textes par catégorie
            if fichier_wc <= 16:
                os.remove(texte)
    return print("fonction nettoyage = ok")

def main(dossier_input=None):
    dossiers_categories = [d for d in Path(dossier_input).iterdir() if d.is_dir()]

    for categorie in dossiers_categories:
        if categorie.name != "Adventure":
            nettoyage_corpus((categorie))
        else:
            shutil.rmtree(categorie)
    
    
            
if __name__ == "__main__":
    args = parser()
    dossier = args.input_dir
    main(dossier)
    