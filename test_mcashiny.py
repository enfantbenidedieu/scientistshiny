

import pandas as pd
from scientisttools import MCA
from scientistshiny import MCAshiny

# Chargement des données
D = pd.read_excel("./data/Data_Methodes_Factorielles.xlsx",sheet_name="ACM_CANINES",index_col=0)
DActives = D[['Taille','Velocite','Affection']]

res_mca1 = MCA(n_components=None,benzecri=False,greenacre=False).fit(DActives)

import pyreadr
result = pyreadr.read_r('./data/poison.rda')
poison = result["poison"]
res_mca2 = MCA(n_components=5,ind_sup=list(range(50,55)),quali_sup = [2,3],quanti_sup =[0,1]).fit(poison)

res_shiny = MCAshiny(model=res_mca2)
res_shiny.run()