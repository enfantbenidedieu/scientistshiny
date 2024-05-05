
import os
import pandas as pd
from scientisttools import PCA
from scientistshiny import PCAshiny
from scientisttools.datasets import load_decathlon2

# PCA with decathlon2
os.chdir("D:/Bureau/PythonProject/packages/scientistshiny/data/")
#decathlon = pd.read_excel("decathlon2.xlsx",header=0,sheet_name=0,index_col=0)
decathlon = load_decathlon2()
res_pca = PCA(standardize=True,ind_sup=list(range(23,27)),quanti_sup=[10,11],quali_sup=12,parallelize=True)
res_pca.fit(decathlon)


# PCA with wine
url = "http://factominer.free.fr/factomethods/datasets/wine.txt"
wine = pd.read_table(url,sep="\t")
res_pca2 = PCA(standardize=True,n_components=5,quanti_sup=[29,30],quali_sup=[0,1],parallelize=True)
res_pca2.fit(wine)

# Chargement des données
D = pd.read_excel("Data_Methodes_Factorielles.xlsx",index_col=0,sheet_name="DATA_ACP_ACTIF")
# ACP normée
res_pca3 = PCA(n_components = None).fit(D)

app = PCAshiny(model=res_pca)
app.run()