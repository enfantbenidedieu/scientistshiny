# -*- coding: utf-8 -*-

# Chargement des librairies
import numpy as np
import pandas as pd
from scientisttools import CA, load_housetasks, load_womenwork, load_children
from scientistshiny import CAshiny

# Model with supplementary elements
D = pd.read_excel("./data/Data_Methodes_Factorielles.xlsx",sheet_name="AFC_FOODS",index_col=0) 
res_ca = CA().fit(D)

housetasks = load_housetasks()
res_ca2 = CA().fit(housetasks)

# Model with supplementary columns
women_work = load_womenwork()
res_ca3 = CA(n_components=None,col_sup=[3,4,5,6]).fit(women_work)

# Model with supplementary rows and columns
children = load_children()
children["group"] = ["A"]*4 + ["B"]*5 + ["C"]*5 +[np.nan]*4
res_ca4 = CA(n_components=None,row_sup=list(range(14,18)),col_sup=list(range(5,8)),quali_sup=8).fit(children)
# Run Shiny App
app = CAshiny(model=res_ca4)
app.run()