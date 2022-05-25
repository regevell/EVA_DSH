# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:38:26 2022

@author: ellio
"""

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
import psychro as psy
import ad_hum_HX
import ad_hum_HXdry
# %matplotlib inline  # uncomment for inline figure
# uncomment for figure in separate window
# %matplotlib qt
# plt.show()

# plt.rcParams["figure.figsize"] = (30,20)
# font = {'weight' : 'bold',
#         'size'   : 30}
# plt.rc('font', **font)

# def RecHX_wd(m=4, β=0.1, θ3=18, φ3=0.5, θ1=-1, φ1=1):
#     θS = 30
#     UA = 935.83
#     from HX import RecHX
#     RecHX(m, β, θS, θ3, φ3, θ1, φ1, UA)

# interact(RecHX_wd, β=(0,0.99,0.1),
#           θSsp = (20,50,2),
#           θ3sp = (17,25,1), φ3sp = (0,1,0.1),
#           θ1 = (-10.,17.,2), φ1 = (0,1,0.1));
m = 4
α = 0.5
β = 0.5
β_HX = 0.5
θS = 18
θIsp = 20
φIsp = 0.3
θO = -1
φO = 0.1
Qsa = 0 #2163.5
Qla = 0 #145.2
mi = 2.12
UA = 935.83
UA_HX = 5000
#x = ad_hum_HX.ModelRecAirmamaHX(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX)
ad_hum_HX.RecAirVAVmamaHX(α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX)
#x = ad_hum_HXdry.ModelRecAirmxmxHXdry(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX)
# def RecHX_wd(m=4, β=0.5, θ3=18, φ3=0.3, θ1=-1, φ1=0.1
#             ):
#     θS = 30
#     UA = 5000
#     x = ModelHXdry(m, β, θS, θ1, φ1, θ3, φ3, UA)
#     if x[3] > psy.w(x[2], 1):
#         RecHX(m, β, θS, θ3, φ3, θ1, φ1, UA)
#         print("Sat HX")
#     else:
#         RecHXdry(m, β, θS, θ3, φ3, θ1, φ1, UA)
#         print("HX dry")


# interact(RecHX_wd, β=(0,0.99,0.1),
#          θSsp = (20,50,2),
#          θ3sp = (17,25,1), φ3sp = (0,1,0.1),
#          θ1 = (-10.,17.,2), φ1 = (0,1,0.1));
