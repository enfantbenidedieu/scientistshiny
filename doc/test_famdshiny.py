# -*- coding: utf-8 -*-
from scientisttools import FAMD, load_autos
from scientistshiny import FAMDshiny
autos = load_autos()

# FAMD with scientistshiny
res_shiny = FAMDshiny(model=autos)
res_shiny.run()

# FAMDshiny on a result of a FAMD
res_famd = FAMD(ind_sup=list(range(35,40)),quanti_sup=[10,11],quali_sup=14,parallelize=False).fit(autos)
res_shiny = FAMDshiny(model=res_famd)
res_shiny.run()