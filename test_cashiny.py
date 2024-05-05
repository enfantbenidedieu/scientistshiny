# -*- coding: utf-8 -*-

# Chargement des librairies
import numpy as np
import pandas as pd
import pyreadr
from scientisttools import CA
from scientistshiny import CAshiny

# Model with supplementary elements
D = pd.read_excel("./data/Data_Methodes_Factorielles.xlsx",sheet_name="AFC_FOODS",index_col=0) 
res_ca = CA().fit(D)

# Model with supplementary columns
url = "http://factominer.free.fr/factomethods/datasets/women_work.txt"
women_work = pd.read_table(url,header=0)
res_ca2 = CA(n_components=None,col_sup=[3,4,5,6]).fit(women_work)

# Model with supplementary rows and columns
result = pyreadr.read_r('./data/children.rda')
children = result["children"]
children["group"] = ["A"]*4 + ["B"]*5 + ["C"]*5 +[np.nan]*4
res_ca3 = CA(n_components=None,row_sup=list(range(14,18)),col_sup=list(range(5,8)),quali_sup=8).fit(children)

#################################################
app = CAshiny(model=res_ca3)
app.run()