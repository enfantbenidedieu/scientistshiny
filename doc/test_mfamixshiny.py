# -*- coding: utf-8 -*-
from scientisttools import MFAMIX, load_gironde
from scientistshiny import MFAMIXshiny
gironde = load_gironde()

# Fill NA by mean in income
gironde.loc[:,"income"] = gironde.loc[:,"income"].fillna(gironde.loc[:,"income"].mean())

name = ["employment","housing","services","environment"] 
group_type = ["s","m","n","s"]

gironde2 = gironde.iloc[:60,:]
res_mfamix = MFAMIX(n_components=5,group=[9,5,9,4],name_group=name,group_type=group_type,ind_sup=list(range(50,60)),num_group_sup=0).fit(gironde2)

res_shiny = MFAMIXshiny(model=res_mfamix)
res_shiny.run()