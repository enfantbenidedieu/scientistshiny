# -*- coding: utf-8 -*-

# Chargement des librairies
import pandas as pd
from scientisttools import CA
from scientistshiny import CAshiny

# Chargement des données
url = "http://factominer.free.fr/factomethods/datasets/women_work.txt"
women_work = pd.read_table(url,header=0)

# Modèle avec colonnes supplémentaires
my_ca = CA(n_components=None,col_sup=[3,4,5,6]).fit(women_work)

import os
os.chdir("d:/Bureau/PythonProject/packages/scientisttools/data/")
D = pd.read_excel("Data_Methodes_Factorielles.xlsx",sheet_name="AFC_FOODS",index_col=0) 

# Modèle sans colonnes supplémentaires

my_ca2 = CA().fit(D)

app = CAshiny(model=my_ca)
app.run()