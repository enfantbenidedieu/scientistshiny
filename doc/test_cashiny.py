# -*- coding: utf-8 -*-
import numpy as np
from scientisttools import CA, load_housetasks, load_children
from scientistshiny import CAshiny

# CA with scientistshiny
housetasks = load_housetasks()
# res_shiny = CAshiny(model=housetasks)
# res_shiny.run()

# CAshiny on a result of a CA
children = load_children()
children["group"] = ["A"]*4 + ["B"]*5 + ["C"]*5 +[np.nan]*4
res_ca2 = CA(n_components=None,row_sup=list(range(14,18)),col_sup=list(range(5,8)),quali_sup=8).fit(children)
res_shiny = CAshiny(model=res_ca2)
res_shiny.run()