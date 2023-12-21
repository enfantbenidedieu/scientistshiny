

import pandas as pd
from scientisttools.decomposition import MCA
from scientistshiny.mca import MCAshiny


href = "D:/Bureau/PythonProject/packages/scientistshiny/data/"

# Chargement des données
D = pd.read_excel(href+"Data_Methodes_Factorielles.xlsx",sheet_name="ACM_CANINES",index_col=0)
DActives = D[['Taille','Velocite','Affection']]

my_mca1 = MCA(n_components=None,
             row_labels=DActives.index,
             var_labels=DActives.columns,
             mod_labels=None,
             matrix_type="completed",
             benzecri=True,
             greenacre=True,
             row_sup_labels=None,
             quali_sup_labels=None,
             quanti_sup_labels=None,
             parallelize=False).fit(DActives)

A= pd.read_excel(href+"races_canines_acm.xlsx",header=0,index_col=0)

my_mca2 = MCA(n_components=None,
             row_labels=A.index[:27],
             var_labels=A.columns[:6],
             mod_labels=None,
             matrix_type="completed",
             benzecri=True,
             greenacre=True,
             row_sup_labels=A.index[27:],
             quali_sup_labels=["Fonction"],
             quanti_sup_labels=["Cote"],
             parallelize=False).fit(A)

res_shiny = MCAshiny(fa_model=my_mca2)
res_shiny.run()