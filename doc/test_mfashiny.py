# -*- coding: utf-8 -*-
from scientisttools import MFA, load_wine, load_gironde
from scientistshiny import MFAshiny
wine = load_wine()
gironde = load_gironde()

# Example 1
group_name = ["origin","odor","visual","odor.after.shaking","taste","overall"]
group = [2,5,3,10,9,2]
num_group_sup = [0,5]
group_type = ["n"]+["s"]*5
res_mfa = MFA(n_components=5,group=group,group_type=group_type,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5]).fit(wine)

# Example 2
name = ["employment","housing","services","environment"] 
group_type = ["s","m","n","s"]
gironde2 = gironde.iloc[:60,:]
res_mfa2 = MFA(n_components=5,group=[9,5,9,4],name_group=name,group_type=group_type,ind_sup=list(range(50,60)),num_group_sup=[1,2]).fit(gironde2)

# MFAshiny on a MFA result
app = MFAshiny(model=res_mfa)
app.run()