# -*- coding: utf-8 -*-
from scientisttools import MFAQUAL, load_poison
from scientistshiny import MFAQUALshiny
poison = load_poison()

group_name = ["desc","desc2","symptom","eat"]
group = [2,2,5,6]
group_type = ["s"]+["n"]*3
num_group_sup = [0,1]

res_mfaqual = MFAQUAL(group=group,name_group=group_name,group_type=group_type,var_weights_mfa=None,num_group_sup=[0,1],ind_sup=list(range(50,55)),parallelize=True).fit(poison)

# Example 2
poison2 = poison.iloc[:,4:]
res_mfaqual2 = MFAQUAL(group=[5,6],name_group=["symptom","eat"],group_type=["n","n"],var_weights_mfa=None,parallelize=True).fit(poison2)

res_shiny = MFAQUALshiny(model=res_mfaqual)
res_shiny.run()