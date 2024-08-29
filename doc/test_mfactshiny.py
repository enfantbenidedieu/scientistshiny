# -*- coding: utf-8 -*-
import pandas as pd
from scientisttools import MFACT, load_mortality
from scientistshiny import MFACTshiny

mortality = load_mortality()
mortality2 = mortality.copy()
mortality2.columns = [x + "-2" for x in mortality2.columns]
dat = pd.concat((mortality,mortality2),axis=1)
res_mfact = MFACT(group=[9]*4,name_group=["1979","2006","1979-2","2006-2"],num_group_sup=[2,3],ind_sup=list(range(50,dat.shape[0])),parallelize=True).fit(dat)

res_shiny = MFACTshiny(model=res_mfact)
res_shiny.run()