

import pandas as pd
from scientisttools.decomposition import PCA
from scientistshiny import PCAshiny

href = "D:/Bureau/PythonProject/packages/scientistshiny/data/"
decathlon = pd.read_excel(href+"/decathlon2.xlsx",header=0,sheet_name=0,index_col=0)

acp = PCA(normalize=True,
          n_components = None,
          row_labels=decathlon.index[:23],
          col_labels=decathlon.columns[:10],
          row_sup_labels=decathlon.index[23:],
          quanti_sup_labels=["Rank","Points"],
          quali_sup_labels=["Competition"],
          parallelize=True).fit(decathlon)


# Chargement des données
D = pd.read_excel(href+"Data_Methodes_Factorielles.xlsx",index_col=0,sheet_name="DATA_ACP_ACTIF")
# ACP normée
acp2 = PCA(normalize=True,
          n_components = None,
          row_labels=D.index,
          col_labels=D.columns,
          row_sup_labels=None,
          quanti_sup_labels=None,
          quali_sup_labels=None,
          parallelize=False).fit(D)

app = PCAshiny(fa_model=acp)
app.run()