#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:56:35 2020
Updated Wed Apr 23 13:20:00 2020
Updated on Sat Apr  2 18:57:50 2022

@author: cghiaus
"""
import numpy as np
import pandas as pd
import psychro as psy
import matplotlib.pyplot as plt

# global variables
# UA = 935.83                 # bldg conductance
# θIsp, wIsp = 18, 6.22e-3    # indoor conditions

θOd = -1                    # outdoor design conditions
mid = 2.18                  # infiltration design

# constants
c = 1e3                     # air specific heat J/kg K
l = 2496e3                  # latent heat J/kg


# *****************************************
# RECYCLED AIR
# *****************************************
def ModelHX(m, β, θS, θ1, φ1, θ3, φ3, UA):
    """
    Model:
        Heating and adiabatic humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations

    INPUTS:
        m       mass flow of supply dry air, kg/s
        α       mixing ratio of outdoor air, -
        β       by-pass factor of the adiabatic humidifier, -
        θS      supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp    indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO      outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latente heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K

    System:
        MX1:    Mixing box
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kθ:     Controller - temperature
        o:      outdoor conditions
        0..5    unknown points (temperature, humidity ratio)

        <----|<-------------------------------------------|
             |                                            |
             |              |-------|                     |
        -o->MX1--0->HC1--1->|       MX2--3->HC2--4->TZ--5-|
                    /       |       |        /      ||    |
                    |       |->AH-2-|        |      BL    |
                    |                        |            |
                    |                        |<-----Kθ----|<-t5
                    |<------------------------------Kw----|<-w5


    Returns
    -------
    x       vector 16 elem.:
            θ0, w0, t1, w1, t2, w2, t3, w3, t4, w4, t5, w5,...
                QHC1, QHC2, QsTZ, QlTZ

    """
    w1 = psy.w(θ1, φ1)            # hum. out
    w3 = psy.w(θ3, φ3)      # indoor mumidity ratio

    # Model
    θs0, Δ_θs = θS, 2             # initial guess saturation temp.

    A = np.zeros((9, 9))          # coefficents of unknowns
    b = np.zeros(9)                # vector of inputs
    while Δ_θs > 0.01:
        # MX1
        A[0, 6], A[0, 7], A[0, 8] = -1, -1, 1
        A[1, 2], A[1, 0], A[1, 6], A[1, 7], b[1] = UA, -UA, 2, 2, UA*(θ3-θ1)
        A[2, 2], A[2, 8], b[2] = m * c, -1, m * c * θ1
        A[3, 0], A[3, 6], b[3] = (1-β) * m * c, 1, (1 - β) * m * c * θ3
        A[4, 1], A[4, 7], b[4] = (1-β) * m * l, 1, (1 - β) * m * l * w3
        A[5, 0], A[5, 1] = psy.wsp(θs0), -1
        b[5] = psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        A[6, 0], A[6, 4], b[6] = - (1 - β), 1, β * θ3
        A[7, 1], A[7, 5], b[7] = - (1 - β), 1, β * w3
        A[8, 3], b[8] = 1, w1

        x = np.linalg.solve(A, b)
        Δ_θs = abs(θs0 - x[4])
        θs0 = x[4]
    return x


def RecHX(m=4, β=0.1, θS=30, θ3=18, φ3=0.49, θ1=-1, φ1=1, UA=935.83):
    """
    Model:
        Heating and adiabatic humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations

    INPUTS:
        α   mixing ratio of outdoor air, -
        β    by-pass factor of the adiabatic humidifier, -
        θS      supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp  indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO    outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latente heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K

    System:
        MX1:    Mixing box
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kθ:     Controller - temperature
        o:      outdoor conditions
        0..5    6 unknown points (temperature, humidity ratio)

        <----|<-------------------------------------------|
             |                                            |
             |              |-------|                     |
        -o->MX1--0->HC1--1->|       MX2--3->HC2--4->TZ--5-|
                    |       |       |        |      ||    |
                    |       |->AH-2-|        |      BL    |
                    |                        |            |
                    |                        |<-----Kθ----|<-t5
                    |<------------------------------Kw----|<-w5

    16 Unknowns
        0..5: 2*6 points (temperature, humidity ratio)
        QsHC1, QsHC2, QsTZ, QlTZ
    Returns
    -------
    None
    """
    plt.close('all')
    w1 = psy.w(θ1, φ1)            # hum. out
    w3 = psy.w(θ3, φ3)
    # Mass flow rate for design conditions
    # Supplay air mass flow rate
    # QsZ = UA*(θO - θIsp) + mi*c*(θO - θIsp)
    # m = - QsZ/(c*(θS - θIsp)
    # where
    # θO, wO = -1, 3.5e-3           # outdoor
    # θS = 30                       # supply air
    # mid = 2.18                     # infiltration
    # QsZ = UA * (θOd - θ3) + mid * c * (θOd - θ3)
    # m = - QsZ / (c * (θS - θ3))
    # m = 4
    print(f'm = {m: 5.3f} kg/s constant for design conditions:')
    print(f'    [θSd = {θS: 3.1f} °C, mi = 2.18 kg/S, θO = -1°C, φ0 = 100%]')

    # Model
    x = ModelHX(m, β, θS, θ1, φ1, θ3, φ3, UA)
    print("Qx = ", x[8])

    θ = np.append(θ1, x[0:5:2])
    w = np.append(w1, x[1:6:2])
    θ = np.append(θ, θ3)
    w = np.append(w, w3)

    # Adjancy matrix
    # Points calc.  1   s   2   4   3       Elements
    # Points pplot  0   1   2   3   4       Elements
    A = np.array([[-1, +0, +1, +0, +0],     # XH
                  [+0, -1, +0, +1, -1]])    # XC

    psy.chartA(θ, w, A)

    θ = pd.Series(θ)
    w = 1000 * pd.Series(w)
    P = pd.concat([θ, w], axis=1)       # points
    P.columns = ['θ [°C]', 'w [g/kg]']

    output = P.to_string(formatters={
        't [°C]': '{:,.2f}'.format,
        'w [g/kg]': '{:,.2f}'.format
    })
    print()
    print(output)

    # Q = pd.Series(x[9:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ'])
    # # Q.columns = ['kW']
    # pd.options.display.float_format = '{:,.2f}'.format
    # print()
    # print(Q.to_frame().T / 1000, 'kW')

    return None


def ModelHXdry(m, β, θS, θ1, φ1, θ3, φ3, UA):

    w1 = psy.w(θ1, φ1)            # hum. out
    w3 = psy.w(θ3, φ3)      # indoor mumidity ratio

    # Model

    A = np.zeros((7, 7))          # coefficents of unknowns
    b = np.zeros(7)                # vector of inputs
    # HX
    A[0, 0], A[0, 2], A[0, 6], b[0] = UA, -UA, 2, UA*(θ3-θ1)
    A[1, 1], b[1] = 1, w1
    A[2, 0], A[2, 6], b[2] = m * c, -1, m * c * θ1
    A[3, 2], A[3, 6], b[3] = (1-β) * m * c, 1, (1 - β) * m * c * θ3
    A[4, 3], b[4] = 1, w3
    A[5, 5], b[5] = 1, w3
    A[6, 2], A[6, 4], b[6] = - (1 - β), 1, β * θ3

    x = np.linalg.solve(A, b)
    return x


def RecHXdry(m, β, θS, θ3, φ3, θ1, φ1, UA):

    plt.close('all')
    w1 = psy.w(θ1, φ1)            # hum. out
    w3 = psy.w(θ3, φ3)
    # Mass flow rate for design conditions
    # Supplay air mass flow rate
    # QsZ = UA*(θO - θIsp) + mi*c*(θO - θIsp)
    # m = - QsZ/(c*(θS - θIsp)
    # where
    # θO, wO = -1, 3.5e-3           # outdoor
    # θS = 30                       # supply air
    # mid = 2.18                     # infiltration
    # QsZ = UA * (θOd - θ3) + mid * c * (θOd - θ3)
    # m = - QsZ / (c * (θS - θ3))
    # m = 4
    print(f'm = {m: 5.3f} kg/s constant for design conditions:')
    print(f'    [θSd = {θS: 3.1f} °C, mi = 2.18 kg/S, θO = -1°C, φ0 = 100%]')

    # Model
    x = ModelHXdry(m, β, θS, θ1, φ1, θ3, φ3, UA)
    print("Qx = ", x[6])

    θ = np.append(θ1, x[0:5:2])
    w = np.append(w1, x[1:6:2])
    θ = np.append(θ, θ3)
    w = np.append(w, w3)

    # Adjancy matrix
    # Points calc.  1   s   2   4   3      Elements
    # Points pplot  0   1   2   3   4       Elements
    A = np.array([[-1, +1, +0, +0, +0],     # XH
                  [+0, +0, -1, +1, -1]])    # XC

    psy.chartA(θ, w, A)

    θ = pd.Series(θ)
    w = 1000 * pd.Series(w)
    P = pd.concat([θ, w], axis=1)       # points
    P.columns = ['θ [°C]', 'w [g/kg]']

    output = P.to_string(formatters={
        't [°C]': '{:,.2f}'.format,
        'w [g/kg]': '{:,.2f}'.format
    })
    print()
    print(output)

    # Q = pd.Series(x[9:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ'])
    # # Q.columns = ['kW']
    # pd.options.display.float_format = '{:,.2f}'.format
    # print()
    # print(Q.to_frame().T / 1000, 'kW')

    return None
