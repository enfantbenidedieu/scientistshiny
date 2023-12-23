
import os
import pandas as pd
from scientisttools.decomposition import FAMD
from scientistshiny.famd import FAMDshiny

os.chdir("d:/Bureau/PythonProject/packages/scientistshiny/data/")
# Chargement des données
D = pd.read_excel("Tennis_Players_AFDM.xlsx",index_col=0)

#instaciation
afdm = FAMD(n_components = None,
            row_labels=list(D.index[0:16]), #jusqu'à Wilander
            row_sup_labels=list(D.index[16:]), #à partir de Djokovic
            quanti_labels=['Taille','Titres','Finales','TitresGC'],
            quanti_sup_labels=['BestClassDouble'],
            quali_labels=['Lateralite','MainsRevers'],
            quali_sup_labels=['RolandGarros'],
            parallelize=False).fit(D)

app = FAMDshiny(fa_model=afdm)
app.run()