# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 12:55:55 2022

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

plt.rcParams["figure.figsize"] = (30,20)
font = {'weight' : 'bold',
        'size'   : 30}
plt.rc('font', **font)

def RecAirVAV_HX_wd(m=3, α=0.5, β=0.1, β_HX=0.3, θS=30, θIsp=18, φIsp=0.5, θO=-1, φO=1):
    Qsa = 0.
    Qla = 0.
    mi = 2.12
    UA = 935.83
    UA_HX = 5000
    m = 3
    from CAV_ad_hum_HX import RecAirVAVmxmxHX, RecAirVAVmxmaHX, RecAirVAVmamxHX, RecAirVAVmamaHX
    from CAV_ad_hum_HXdry import RecAirVAVmxmxHXdry, RecAirVAVmxmaHXdry, RecAirVAVmamxHXdry, RecAirVAVmamaHXdry
    
    x = RecAirVAVmxmxHXdry(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, True)
    
    if x[3] > psy.w(x[2], 1):
        x = RecAirVAVmamxHXdry(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, True)
        if x[11] > psy.w(x[10], 1):
            # mamaHXdry
            x = RecAirVAVmamaHXdry(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, True)
            if x[19] > psy.w(x[18], 1):  # dry or not
                RecAirVAVmamaHX(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, False)
                print('mamaHX')
            else:
                RecAirVAVmamaHXdry(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, False)
                print('mamaHXdry')
        else:
            # mamxHXdry
            if x[17] > psy.w(x[16], 1):
                RecAirVAVmamxHX(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, False)
                print('mamxHX')
            else:
                RecAirVAVmamxHXdry(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, False)
                print('mamxHXdry')
    elif x[9] > psy.w(x[8], 1):
        # mxmaHXdry
        x = RecAirVAVmxmaHXdry(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, True)
        if x[17] > psy.w(x[16], 1):
            RecAirVAVmxmaHX(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, False)
            print('mxmaHX')
        else:
            RecAirVAVmxmaHXdry(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, False)
            print('mxmaHXdry')
    else:
        # mxmxHXdry
        if x[15] > psy.w(x[14], 1):
            RecAirVAVmxmxHX(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, False)
            print('mxmxHX')
        else:
            RecAirVAVmxmxHXdry(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX, False)
            print('mxmxHXdry')
            
interact(RecAirVAV_HX_wd, m=(0.2, 6, 0.2), α=(0, 1, 0.1), β=(0, 0.99, 0.1), β_HX=(0, 0.99, 0.1), θSsp=(20, 50, 2),
          θIsp=(17, 25, 1), φIsp=(0, 1, 0.1),
          θO = (-10., 17., 2), φO = (0, 1, 0.1));