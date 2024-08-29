# -*- coding: utf-8 -*-
from scientisttools import PCA, load_decathlon2, load_wine
from scientistshiny import PCAshiny

# PCA with scientistshiny
decathlon = load_decathlon2()
res_shiny = PCAshiny(model=decathlon)
res_shiny.run()

## PCAshiny on a result of a PCA
# Example 1 - decathlon dataset
res_pca1 = PCA(standardize=True,ind_sup=list(range(23,27)),quanti_sup=[10,11],quali_sup=12,parallelize=True).fit(decathlon)
# Example 2 - wine dataset
wine = load_wine()
res_pca2 = PCA(standardize=True,n_components=5,quanti_sup=[29,30],quali_sup=[0,1],parallelize=True).fit(wine)
res_shiny = PCAshiny(model=res_pca1)
res_shiny.run()