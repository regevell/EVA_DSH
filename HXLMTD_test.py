# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:38:26 2022

@author: ellio
"""

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
import psychro as psy
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

def RecHX_wd(m=4, β=0.5, θ3=18, φ3=0.3, θ1=-1, φ1=0.1
            ):
    θS = 30
    UA = 5000
    from HXLMTD import RecHXdry
    from HXLMTD import ModelHXdry
    from HXLMTD import RecHX
    x = ModelHXdry(m, β, θS, θ1, φ1, θ3, φ3, UA)
    if x[3] > psy.w(x[2], 1):
        RecHX(m, β, θS, θ3, φ3, θ1, φ1, UA)
        print("Sat HX")
    else:
        RecHXdry(m, β, θS, θ3, φ3, θ1, φ1, UA)
        print("HX dry")


interact(RecHX_wd, β=(0,0.99,0.1),
         θSsp = (20,50,2),
         θ3sp = (17,25,1), φ3sp = (0,1,0.1),
         θ1 = (-10.,17.,2), φ1 = (0,1,0.1));
