#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 2020
Last Updated on Tue May 17 2022

@authors: cghiaus, lbeber, eregev, cgerike-roberts
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
def ModelRecAirmxmx(m, α, β, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA):
    """
    Model:
        Heating and adiabatic humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations
        mixing-points after both mixers are below the saturation curve

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
            θ0, w0, t1, w1, t2, w2, t3, w3, t4, w4, t5, w5,
                QHC1, QHC2, QsTZ, QlTZ

    """
    Kθ, Kw = 1e10, 1e10           # controller gain
    wO = psy.w(θO, φO)            # hum. out
    wIsp = psy.w(θIsp, φIsp)      # indoor humidity ratio

    # Model
    θs0, Δ_θs = θS, 2             # initial guess saturation temp.

    A = np.zeros((16, 16))          # coefficients of unknowns
    b = np.zeros(16)                # vector of inputs
    while Δ_θs > 0.01:
        # MX1
        A[0, 0], A[0, 10], b[0] = m * c, -(1 - α) * m * c, α * m * c * θO
        A[1, 1], A[1, 11], b[1] = m * l, -(1 - α) * m * l, α * m * l * wO
        # HC1
        A[2, 0], A[2, 2], A[2, 12] = m * c, -m * c, 1
        A[3, 1], A[3, 3]= m * l, -m * l
        # AH
        A[4, 2], A[4, 3], A[4, 4], A[4, 5]= c, l, -c, -l
        A[5, 4], A[5, 5] = psy.wsp(θs0), -1
        b[5] = psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # MX2
        A[6, 2], A[6, 4], A[6, 6] = β * m * c, (1 - β) * m * c, -m * c
        A[7, 3], A[7, 5], A[7, 7] = β * m * l, (1 - β) * m * l, -m * l
        # HC2
        A[8, 6], A[8, 8], A[8, 13] = m * c, -m * c, 1
        A[9, 7], A[9, 9] = m * l, -m * l
        # TZ
        A[10, 8], A[10, 10], A[10, 14] = m * c, -m * c, 1
        A[11, 9], A[11, 11], A[11, 15] = m * l, -m * l, 1
        # BL
        A[12, 10], A[12, 14], b[12] = (UA + mi * c), 1, (UA + mi * c) * θO + Qsa
        A[13, 11], A[13, 15], b[13] = mi * l, 1, mi * l * wO + Qla
        # Kθ & Kw
        A[14, 10], A[14, 12], b[14] = Kθ, 1, Kθ * θIsp
        A[15, 11], A[15, 13], b[15] = Kw, 1, Kw * wIsp

        x = np.linalg.solve(A, b)
        Δ_θs = abs(θs0 - x[4])
        θs0 = x[4]
    return x


def ModelRecAirmxma(m, α, β, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA):
    """
    Model:
        Heating and adiabatic humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations
        mixing-point after first mixer is below the saturation curve
        mixing-point after second mixer is above the saturation curve

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
        MX_AD2: Adiabatic humidification/condensation
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kθ:     Controller - temperature
        o:      outdoor conditions
        0..6    unknown points (temperature, humidity ratio)

        <----|<------------------------------------------------------|
             |                                                       |
             |              |-------|                                |
        -o->MX1--0->HC1--1->|       MX2--3->MX_AD2--4->HC2--5->TZ--6-|
                    /       |       |                   /      ||    |
                    |       |->AH-2-|                   |      BL    |
                    |                                   |            |
                    |                                   |<-----Kθ----|<-t6
                    |<-----------------------------------------Kw----|<-w6
    Returns
    -------
    x       vector 18 elem.:
            θ0, w0, t1, w1, t2, w2, t3, w3, t4, w4, t5, w5, t6, w6,
                QHC1, QHC2, QsTZ, QlTZ

    """
    Kθ, Kw = 1e10, 1e10           # controller gain
    wO = psy.w(θO, φO)            # hum. out
    wIsp = psy.w(θIsp, φIsp)      # indoor humidity ratio

    # Model
    θs0, Δ_θs = θS, 2             # initial guess saturation temp.

    A = np.zeros((18, 18))          # coefficients of unknowns
    b = np.zeros(18)                # vector of inputs
    while Δ_θs > 0.01:
        # MX1
        A[0, 0], A[0, 12], b[0] = m * c, -(1 - α) * m * c, α * m * c * θO
        A[1, 1], A[1, 13], b[1] = m * l, -(1 - α) * m * l, α * m * l * wO
        # HC1
        A[2, 0], A[2, 2], A[2, 14] = m * c, -m * c, 1
        A[3, 1], A[3, 3] = m * l, -m * l
        # AH
        A[4, 2], A[4, 3], A[4, 4], A[4, 5] = c, l, -c, -l
        A[5, 4], A[5, 5] = psy.wsp(θs0), -1
        b[5] = psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # MX2
        A[6, 2], A[6, 4], A[6, 6] = β * m * c, (1 - β) * m * c, -m * c
        A[7, 3], A[7, 5], A[7, 7] = β * m * l, (1 - β) * m * l, -m * l
        # MX_AD2
        A[8, 6], A[8, 7], A[8, 8], A[8, 9] = c, l, -c, -l
        A[9, 8], A[9, 9], b[9] = psy.wsp(θs0), -1, psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # HC2
        A[10, 8], A[10, 10], A[10, 15] = m * c, -m * c, 1
        A[11, 9], A[11, 11] = m * l, -m * l
        # TZ
        A[12, 10], A[12, 12], A[12, 16] = m * c, -m * c, 1
        A[13, 11], A[13, 13], A[13, 17] = m * l, -m * l, 1
        # BL
        A[14, 13], A[14, 16], b[14] = (UA + mi * c), 1, (UA + mi * c) * θO + Qsa
        A[15, 13], A[15, 17], b[15] = mi * l, 1, mi * l * wO + Qla
        # Kθ & Kw
        A[16, 12], A[16, 14], b[16] = Kθ, 1, Kθ * θIsp
        A[17, 13], A[17, 15], b[17] = Kw, 1, Kw * wIsp

        x = np.linalg.solve(A, b)
        Δ_θs = abs(θs0 - x[4])
        θs0 = x[4]
    return x


def ModelRecAirmamx(m, α, β, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA):
    """
    Model:
        Heating and adiabatic humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations
        mixing-point after first mixer is above the saturation curve
        mixing-point after second mixer is below the saturation curve

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
        MX_AD1: Adiabatic humidification/condensation
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kθ:     Controller - temperature
        o:      outdoor conditions
        0..6    unknown points (temperature, humidity ratio)

        <----|<---------------------------------------------------------|
             |                                                          |
             |                            |-------|                     |
        -o->MX1--0->MX_AD1 --1--->HC1--2->|       MX2--4->HC2--5->TZ--6-|
                                  /       |       |        /      ||    |
                                  |       |->AH-3-|        |      BL    |
                                  |                        |            |
                                  |                        |<-----Kθ----|<-t6
                                  |<------------------------------Kw----|<-w6


    Returns
    -------
    x       vector 18 elem.:
            θ0, w0, t1, w1, t2, w2, t3, w3, t4, w4, t5, w5, t6, w6,
                QHC1, QHC2, QsTZ, QlTZ

    """
    Kθ, Kw = 1e10, 1e10  # controller gain
    wO = psy.w(θO, φO)  # hum. out
    wIsp = psy.w(θIsp, φIsp)  # indoor humidity ratio

    # Model
    θs0, Δ_θs = θS, 2  # initial guess saturation temp.

    A = np.zeros((18, 18))  # coefficients of unknowns
    b = np.zeros(18)  # vector of inputs
    while Δ_θs > 0.01:
        # MX1
        A[0, 0], A[0, 12], b[0] = m * c, -(1 - α) * m * c, α * m * c * θO
        A[1, 1], A[1, 13], b[1] = m * l, -(1 - α) * m * l, α * m * l * wO
        # MX_AD1
        A[2, 0], A[2, 1], A[2, 2], A[2, 3] = c, l, -c, -l
        A[3, 2], A[3, 3], b[3] = psy.wsp(θs0), -1, psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # HC1
        A[4, 2], A[4, 4], A[4, 16] = m * c, -m * c, 1
        A[5, 3], A[5, 5] = m * l, -m * l
        # AH
        A[6, 4], A[6, 5], A[6, 6], A[6, 7] = c, l, -c, -l
        A[7, 6], A[7, 7] = psy.wsp(θs0), -1
        b[7] = psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # MX2
        A[8, 4], A[8, 6], A[8, 8] = β * m * c, (1 - β) * m * c, -m * c
        A[9, 5], A[9, 7], A[9, 9] = β * m * l, (1 - β) * m * l, -m * l
        # HC2
        A[10, 8], A[10, 10], A[10, 15] = m * c, -m * c, 1
        A[11, 9], A[11, 11] = m * l, -m * l
        # TZ
        A[12, 10], A[12, 12], A[12, 16] = m * c, -m * c, 1
        A[13, 11], A[13, 13], A[13, 17] = m * l, -m * l, 1
        # BL
        A[14, 13], A[14, 16], b[14] = (UA + mi * c), 1, (UA + mi * c) * θO + Qsa
        A[15, 13], A[15, 17], b[15] = mi * l, 1, mi * l * wO + Qla
        # Kθ & Kw
        A[16, 12], A[16, 14], b[16] = Kθ, 1, Kθ * θIsp
        A[17, 13], A[17, 15], b[17] = Kw, 1, Kw * wIsp

        x = np.linalg.solve(A, b)
        Δ_θs = abs(θs0 - x[6])
        θs0 = x[6]
    return x


def ModelRecAirmama(m, α, β, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA):
    """
    Model:
        Heating and adiabatic humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations
        mixing-points after both mixers are above the saturation curve

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
        0..7    unknown points (temperature, humidity ratio)


<----|<-----------------------------------------------------------------|
     |                                                                  |
     |                         |-------|                                |
-o->MX1--0->MX_AD1--1->HC1--2->|       MX2--4->MX_AD2--5->HC2--5->TZ--6-|
                       /       |       |                  /      ||     |
                       |       |->AH-3-|                  |      BL     |
                       |                                  |             |
                       |                                  |<-----Kθ-----|<-t7
                       |<----------------------------------------Kw-----|<-w7

    Returns
    -------
    x       vector 20 elem.:
            θ0, w0, t1, w1, t2, w2, t3, w3, t4, w4, t5, w5, t6, w6, t7, w7,
                QHC1, QHC2, QsTZ, QlTZ

    """
    Kθ, Kw = 1e10, 1e10  # controller gain
    wO = psy.w(θO, φO)  # hum. out
    wIsp = psy.w(θIsp, φIsp)  # indoor humidity ratio

    # Model
    θs0, Δ_θs = θS, 2  # initial guess saturation temp.

    A = np.zeros((20, 20))  # coefficients of unknowns
    b = np.zeros(20)  # vector of inputs
    while Δ_θs > 0.01:
        # MX1
        A[0, 0], A[0, 12], b[0] = m * c, -(1 - α) * m * c, α * m * c * θO
        A[1, 1], A[1, 13], b[1] = m * l, -(1 - α) * m * l, α * m * l * wO
        # MX_AD1
        A[2, 0], A[2, 1], A[2, 2], A[2, 3] = c, l, -c, -l
        A[3, 2], A[3, 3], b[3] = psy.wsp(θs0), -1, psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # HC1
        A[4, 2], A[4, 4], A[4, 16] = m * c, -m * c, 1
        A[5, 3], A[5, 5] = m * l, -m * l
        # AH
        A[6, 4], A[6, 5], A[6, 6], A[6, 7] = c, l, -c, -l
        A[7, 6], A[7, 7] = psy.wsp(θs0), -1
        b[7] = psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # MX2
        A[8, 4], A[8, 6], A[8, 8] = β * m * c, (1 - β) * m * c, -m * c
        A[9, 5], A[9, 7], A[9, 9] = β * m * l, (1 - β) * m * l, -m * l
        # MX_AD1
        A[10, 8], A[10, 9], A[10, 10], A[10, 11] = c, l, -c, -l
        A[11, 10], A[11, 11], b[11] = psy.wsp(θs0), -1, psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # HC2
        A[12, 10], A[12, 12], A[12, 17] = m * c, -m * c, 1
        A[13, 11], A[13, 13] = m * l, -m * l
        # TZ
        A[14, 12], A[14, 14], A[14, 18] = m * c, -m * c, 1
        A[15, 13], A[15, 15], A[15, 19] = m * l, -m * l, 1
        # BL
        A[16, 14], A[16, 18], b[16] = (UA + mi * c), 1, (UA + mi * c) * θO + Qsa
        A[17, 15], A[17, 19], b[17] = mi * l, 1, mi * l * wO + Qla
        # Kθ & Kw
        A[18, 14], A[18, 16], b[18] = Kθ, 1, Kθ * θIsp
        A[19, 15], A[19, 17], b[19] = Kw, 1, Kw * wIsp

        x = np.linalg.solve(A, b)
        Δ_θs = abs(θs0 - x[6])
        θs0 = x[6]
    return x


def RecAirVAVmxmx(α=1, β=0.1,
                  θSsp=30, θIsp=18, φIsp=0.49, θO=-1, φO=1,
                  Qsa=0, Qla=0, mi=2.18, UA=935.83, check=True):
    """
    Heating & Adiabatic humidification & Re-heating
    Recirculated air
    VAV Variable Air Volume:
        mass flow rate calculated to have const. supply temp.

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
        0..5    unknown points (temperature, humidity ratio)

        <----|<-------------------------------------------------|
             |                                                  |
             |              |-------|                           |
        -o->MX1--0->HC1--1->|       MX2--3->HC2--F-----4->TZ--5-|
                    |       |->AH-2-|        |   |     |  ||    |
                    |                        |   |-Kθ4-|  BL    |
                    |                        |                  |
                    |                        |<-----Kθ----------|<-t5
                    |<------------------------------Kw----------|<-w5

    16 Unknowns
        0..5: 2*6 points (temperature, humidity ratio)
        QsHC1, QsHC2, QsTZ, QlTZ
    """
    from scipy.optimize import least_squares

    def Saturation(m):
        """
        Used in VAV to find the mass flow which solves θS = θSsp
        Parameters
        ----------
            m : mass flow rate of dry air

        Returns
        -------
            θS - θSsp: difference between supply temp. and its set point

        """
        x = ModelRecAirmxmx(m, α, β,
                        θSsp, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA)
        θS = x[8]
        return (θS - θSsp)

    plt.close('all')
    wO = psy.w(θO, φO)            # hum. out

    # Mass flow rate
    res = least_squares(Saturation, 5, bounds=(0, 10))
    if res.cost < 1e-10:
        m = float(res.x)
    else:
        print('RecAirVAV: No solution for m')

    print(f'm = {m: 5.3f} kg/s')
    x = ModelRecAirmxmx(m, α, β, θSsp, θIsp, φIsp,
                        θO, φO, Qsa, Qla, mi, UA)
    if not check:
        θ = np.append(θO, x[0:12:2])
        w = np.append(wO, x[1:12:2])

        # Adjacency matrix
        # Points calc.  o   0   1   2   3   4   5       Elements
        # Points plot   0   1   2   3   4   5   6       Elements
        A = np.array([[-1, +1, +0, +0, +0, +0, -1],     # MX1
                      [+0, -1, +1, +0, +0, +0, +0],     # HC1
                      [+0, +0, -1, +1, +0, +0, +0],     # AH
                      [+0, +0, -1, -1, +1, +0, +0],     # MX2
                      [+0, +0, +0, +0, -1, +1, +0],     # HC2
                      [+0, +0, +0, +0, +0, -1, +1]])    # TZ

        psy.chartA(θ, w, A)

        θ = pd.Series(θ)
        w = 1000 * pd.Series(w)
        P = pd.concat([θ, w], axis=1)       # points
        P.columns = ['θ [°C]', 'w [g/kg]']

        output = P.to_string(formatters={
            'θ [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format
        })
        print()
        print(output)

        Q = pd.Series(x[12:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ'])
        # Q.columns = ['kW']
        pd.options.display.float_format = '{:,.2f}'.format
        print()
        print(Q.to_frame().T / 1000, 'kW')

        return None
    else:
        return x




def RecAirVAVmxma(α=1, β=0.1,
                  θSsp=30, θIsp=18, φIsp=0.49, θO=-1, φO=1,
                  Qsa=0, Qla=0, mi=2.18, UA=935.83, check=True):
    """
    Heating & Adiabatic Mixing at second Mixer & Adiabatic humidification & Re-heating
    Recirculated air
    VAV Variable Air Volume:
        mass flow rate calculated to have const. supply temp.

    INPUTS:
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
        MX_AD2: Adiabatic humidification/condensation
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kθ:     Controller - temperature
        o:      outdoor conditions
        0..6    unknown points (temperature, humidity ratio)


        <----|<------------------------------------------------------------|
             |                                                             |
             |              |-------|                                      |
        -o->MX1--0->HC1--1->|       MX2--3->MX_AD2--4->HC2--F-----5->TZ--6-|
                    /       |       |                   |   |     |  ||    |
                    |       |->AH-2-|                   |   |-Kθ5-|  BL    |
                    |                                   |                  |
                    |                                   |<-----Kθ----------|<-t6
                    |<-----------------------------------------Kw----------|<-w6

    18 Unknowns
        0..6: 2*7 points (temperature, humidity ratio)
        QsHC1, QsHC2, QsTZ, QlTZ
    """
    from scipy.optimize import least_squares

    def Saturation(m):
        """
        Used in VAV to find the mass flow which solves θS = θSsp
        Parameters
        ----------
            m : mass flow rate of dry air

        Returns
        -------
            θS - θSsp: difference between supply temp. and its set point

        """
        x = ModelRecAirmxma(m, α, β, θSsp, θIsp, φIsp,
                            θO, φO, Qsa, Qla, mi, UA)
        θS = x[10]
        return (θS - θSsp)

    plt.close('all')
    wO = psy.w(θO, φO)            # hum. out

    # Mass flow rate
    res = least_squares(Saturation, 5, bounds=(0, 10))
    if res.cost < 1e-10:
        m = float(res.x)
    else:
        print('RecAirVAV: No solution for m')

    print(f'm = {m: 5.3f} kg/s')
    x = ModelRecAirmxma(m, α, β, θSsp, θIsp, φIsp,
                        θO, φO, Qsa, Qla, mi, UA)

    if not check:
        θ = np.append(θO, x[0:14:2])
        w = np.append(wO, x[1:14:2])

        # Adjacency matrix
        # Points calc.  o   0   1   2   3   4   5   6       Elements
        # Points plot   0   1   2   3   4   5   6   7       Elements
        A = np.array([[-1, +1, +0, +0, +0, +0, +0, -1],     # MX1
                      [+0, -1, +1, +0, +0, +0, +0, +0],     # HC1
                      [+0, +0, -1, +1, +0, +0, +0, +0],     # AH
                      [+0, +0, -1, -1, +1, +0, +0, +0],     # MX2
                      [+0, +0, +0, +0, -1, +1, +0, +0],     # MX_AD2
                      [+0, +0, +0, +0, +0, -1, +1, +0],     # HC2
                      [+0, +0, +0, +0, +0, +0, -1, +1]])    # TZ

        psy.chartA(θ, w, A)

        θ = pd.Series(θ)
        w = 1000 * pd.Series(w)
        P = pd.concat([θ, w], axis=1)       # points
        P.columns = ['θ [°C]', 'w [g/kg]']

        output = P.to_string(formatters={
            'θ [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format
        })
        print()
        print(output)

        Q = pd.Series(x[14:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ'])
        # Q.columns = ['kW']
        pd.options.display.float_format = '{:,.2f}'.format
        print()
        print(Q.to_frame().T / 1000, 'kW')

        return None
    else:
        return x


def RecAirVAVmamx(α=1, β=0.1,
                  θSsp=30, θIsp=18, φIsp=0.49, θO=-1, φO=1,
                  Qsa=0, Qla=0, mi=2.18, UA=935.83, check=True):
    """
    Heating & Adiabatic Mixing at first Mixer & Adiabatic humidification & Re-heating
    Recirculated air
    VAV Variable Air Volume:
        mass flow rate calculated to have const. supply temp.

    INPUTS:
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
        MX_AD1: Adiabatic humidification/condensation
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kθ:     Controller - temperature
        o:      outdoor conditions
        0..6    unknown points (temperature, humidity ratio)

        <----|<------------------------------------------------------------|
             |                                                             |
             |                         |-------|                           |
        -o->MX1--0->MX_AD1--1->HC1--2->|       MX2--4->HC2--F-----5->TZ--6-|
                    /                  |       |        |   |     |  ||    |
                    |                  |->AH-3-|        |   |-Kθ5-|  BL    |
                    |                                   |                  |
                    |                                   |<-----Kθ----------|<-t6
                    |<-----------------------------------------Kw----------|<-w6

    18 Unknowns
        0..6: 2*7 points (temperature, humidity ratio)
        QsHC1, QsHC2, QsTZ, QlTZ
    """
    from scipy.optimize import least_squares

    def Saturation(m):
        """
        Used in VAV to find the mass flow which solves θS = θSsp
        Parameters
        ----------
            m : mass flow rate of dry air

        Returns
        -------
            θS - θSsp: difference between supply temp. and its set point

        """
        x = ModelRecAirmamx(m, α, β, θSsp, θIsp, φIsp,
                            θO, φO, Qsa, Qla, mi, UA)
        θS = x[10]
        return (θS - θSsp)

    plt.close('all')
    wO = psy.w(θO, φO)            # hum. out

    # Mass flow rate
    res = least_squares(Saturation, 5, bounds=(0, 10))
    if res.cost < 1e-5:
        m = float(res.x)
    else:
        print('RecAirVAV: No solution for m')

    print(f'm = {m: 5.3f} kg/s')
    x = ModelRecAirmamx(m, α, β, θSsp, θIsp, φIsp,
                        θO, φO, Qsa, Qla, mi, UA)

    if not check:
        θ = np.append(θO, x[0:14:2])
        w = np.append(wO, x[1:14:2])

        # Adjacency matrix
        # Points calc.  o   0   1   2   3   4   5   6       Elements
        # Points plot   0   1   2   3   4   5   6   7       Elements
        A = np.array([[-1, +1, +0, +0, +0, +0, +0, -1],     # MX1
                      [+0, -1, +1, +0, +0, +0, +0, +0],     # MX_AD1
                      [+0, +0, -1, +1, +0, +0, +0, +0],     # HC1
                      [+0, +0, +0, -1, +1, +0, +0, +0],     # AH
                      [+0, +0, +0, -1, -1, +1, +0, +0],     # MX2
                      [+0, +0, +0, +0, +0, -1, +1, +0],     # HC2
                      [+0, +0, +0, +0, +0, +0, -1, +1]])    # TZ

        psy.chartA(θ, w, A)

        θ = pd.Series(θ)
        w = 1000 * pd.Series(w)
        P = pd.concat([θ, w], axis=1)       # points
        P.columns = ['θ [°C]', 'w [g/kg]']

        output = P.to_string(formatters={
            'θ [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format
        })
        print()
        print(output)

        Q = pd.Series(x[14:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ'])
        # Q.columns = ['kW']
        pd.options.display.float_format = '{:,.2f}'.format
        print()
        print(Q.to_frame().T / 1000, 'kW')

        return None
    else:
        return x


def RecAirVAVmama(α=1, β=0.1,
                  θSsp=30, θIsp=18, φIsp=0.49, θO=-1, φO=1,
                  Qsa=0, Qla=0, mi=2.18, UA=935.83, check=True):
    """
    Heating & Adiabatic Mixing at both Mixers & Adiabatic humidification & Re-heating
    Recirculated air
    VAV Variable Air Volume:
        mass flow rate calculated to have const. supply temp.

    INPUTS:
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
        MX_AD1: Adiabatic humidification/condensation
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        MX_AD2: Adiabatic humidification/condensation
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kθ:     Controller - temperature
        o:      outdoor conditions
        0..7    unknown points (temperature, humidity ratio)

        <----|<------------------------------------------------------------|
             |                                                             |
             |                         |-------|                           |
        -o->MX1--0->MX_AD1--1->HC1--2->|       MX2--4->MX_AD1--5->HC2--F-----6->TZ--7-|
                    /                  |       |                   |   |     |  ||    |
                    |                  |->AH-3-|                   |   |-Kθ6-|  BL    |
                    |                                              |                  |
                    |                                              |<-----Kθ----------|<-t7
                    |<----------------------------------------------------Kw----------|<-w7

    20 Unknowns
        0..7: 2*8 points (temperature, humidity ratio)
        QsHC1, QsHC2, QsTZ, QlTZ
    """
    from scipy.optimize import least_squares

    def Saturation(m):
        """
        Used in VAV to find the mass flow which solves θS = θSsp
        Parameters
        ----------
            m : mass flow rate of dry air

        Returns
        -------
            θS - θSsp: difference between supply temp. and its set point

        """
        x = ModelRecAirmama(m, α, β, θSsp, θIsp, φIsp,
                            θO, φO, Qsa, Qla, mi, UA)
        θS = x[12]
        return (θS - θSsp)

    plt.close('all')
    wO = psy.w(θO, φO)            # hum. out

    # Mass flow rate
    res = least_squares(Saturation, 5, bounds=(0, 10))
    if res.cost < 1e-10:
        m = float(res.x)
    else:
        print('RecAirVAV: No solution for m')

    print(f'm = {m: 5.3f} kg/s')
    x = ModelRecAirmama(m, α, β, θSsp, θIsp, φIsp,
                        θO, φO, Qsa, Qla, mi, UA)

    if not check:
        θ = np.append(θO, x[0:16:2])
        w = np.append(wO, x[1:16:2])

        # Adjacency matrix
        # Points calc.  o   0   1   2   3   4   5   6   7       Elements
        # Points plot   0   1   2   3   4   5   6   7   8       Elements
        A = np.array([[-1, +1, +0, +0, +0, +0, +0, +0, -1],     # MX1
                      [+0, -1, +1, +0, +0, +0, +0, +0, +0],     # MX_AD1
                      [+0, +0, -1, +1, +0, +0, +0, +0, +0],     # HC1
                      [+0, +0, +0, -1, +1, +0, +0, +0, +0],     # AH
                      [+0, +0, +0, -1, -1, +1, +0, +0, +0],     # MX2
                      [+0, +0, +0, +0, +0, -1, +1, +0, +0],     # MX_AD2
                      [+0, +0, +0, +0, +0, +0, -1, +1, +0],     # HC2
                      [+0, +0, +0, +0, +0, +0, +0, -1, +1]])    # TZ

        psy.chartA(θ, w, A)

        θ = pd.Series(θ)
        w = 1000 * pd.Series(w)
        P = pd.concat([θ, w], axis=1)       # points
        P.columns = ['θ [°C]', 'w [g/kg]']

        output = P.to_string(formatters={
            'θ [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format
        })
        print()
        print(output)

        Q = pd.Series(x[16:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ'])
        # Q.columns = ['kW']
        pd.options.display.float_format = '{:,.2f}'.format
        print()
        print(Q.to_frame().T / 1000, 'kW')

        return None
    else:
        return x