
import os
import pandas as pd
from scientisttools import PCA
from scientistshiny import PCAshiny
from scientisttools import load_decathlon2, load_wine

# Chargement des données
D = pd.read_excel("./data/Data_Methodes_Factorielles.xlsx",index_col=0,sheet_name="DATA_ACP_ACTIF")
# ACP normée
res_pca = PCA(n_components = None).fit(D)

# PCA with wine
wine = load_wine()
res_pca2 = PCA(standardize=True,n_components=5,quanti_sup=[29,30],quali_sup=[0,1],parallelize=True).fit(wine)

# PCA with decathlon2
decathlon = load_decathlon2()
res_pca3 = PCA(standardize=True,ind_sup=list(range(23,27)),quanti_sup=[10,11],quali_sup=12,parallelize=True).fit(decathlon)

app = PCAshiny(model=res_pca3)
app.run()