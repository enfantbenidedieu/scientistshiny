# -*- coding: utf-8 -*-

# Chargement des librairies
import pandas as pd
from scientisttools.decomposition import CA
from scientistshiny import CAshiny

# Chargement des données
url = "http://factominer.free.fr/factomethods/datasets/women_work.txt"
women_work = pd.read_table(url,header=0)

# Modèle avec colonnes supplémentaires
my_ca1 = CA(n_components=None,
            row_labels=women_work.index.values,
            col_labels=women_work.columns[:3].values,
            row_sup_labels=None,
            col_sup_labels=women_work.columns[3:].values).fit(women_work)

import os
os.chdir("d:/Bureau/PythonProject/packages/scientisttools/data/")
D = pd.read_excel("Data_Methodes_Factorielles.xlsx",sheet_name="AFC_FOODS",index_col=0) 

# Modèle sans colonnes supplémentaires
my_ca2 = CA(n_components=None,
           row_labels=D.index,
           col_labels=D.columns,
           row_sup_labels=None,
           col_sup_labels=None).fit(D)

app = CAshiny(fa_model=my_ca1)
app.run()