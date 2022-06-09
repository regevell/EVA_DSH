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

θOd = -1  # outdoor design conditions
mid = 2.18  # infiltration design

# constants
c = 1e3  # air specific heat J/kg K
l = 2496e3  # latent heat J/kg


# *****************************************
# RECYCLED AIR
# *****************************************
def ModelRecAirmxmxHX(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX):
    """
    Model:
        Heat Exchanger
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
        β_HX    by-pass factor of the eat exchanger, -
        θS      supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp    indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO      outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latente heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K
        UA_HX   global conductivity HX, W/K
    System:
        XH:     Heating in Heat Exchanger
        MX1:    Mixing box
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kθ:     Controller - temperature
        XC:     Cooling in Heat Exchanger
        XM:     Mixing in Heat Exchanger
        Qs:     Exchanged sensible Heat
        Ql:     Exchanged latent Heat
        o:      outdoor conditions
        0..8    unknown points (temperature, humidity ratio)

        |--------|
    <-8-XM       |<---|<------------------------------------------|
        |        |    |                                           |
        |<-7-XC--|    |                                           |
            |  |      |                                           |
            Qs+Ql     |                                           |
             |        /             |-------|                     |
        --o->XH--0->MX1--1->HC1--2->|       MX2--4->HC2--5->TZ--6-|
                            /       |       |        /      ||    |
                            |       |->AH-3-|        |      BL    |
                            |                        |            |
                            |                        |<-----Kθ----|<-t6
                            |<------------------------------Kw----|<-w6


    Returns
    -------
    x       vector 25 elem.:
            θ0, w0, θ1, w1, θ2, w2, θ3, w3, θ4, w4, θ5, w5, θ6, w6, θ7, w7, θ8, w8,
            QHC1, QHC2, QsTZ, QlTZ, Qx, Qs, Ql

    """
    Kθ, Kw = 1e10, 1e10  # controller gain
    wO = psy.w(θO, φO)  # hum. out
    wIsp = psy.w(θIsp, φIsp)  # indoor humidity ratio

    # Model
    θs0, Δ_θs0 = θS, 2  # initial guess saturation temp.
    θs1, Δ_θs1 = θS, 2  # initial guess saturation temp.
    
    A = np.zeros((25, 25))  # coefficients of unknowns
    b = np.zeros(25)  # vector of inputs
    while Δ_θs0 > 0.01 or Δ_θs1 > 0.01:
        # XH
        A[0, 0], A[0, 22], b[0] = -m * c, 1, -m * c * θO
        A[1, 1], b[1] = 1, wO
        # MX1
        A[2, 0], A[2, 2], A[2, 12] = α * m * c, -m * c, (1 - α) * m * c
        A[3, 1], A[3, 3], A[3, 13] = α * m * l, -m * l, (1 - α) * m * l
        # HC1
        A[4, 2], A[4, 4], A[4, 18] = m * c, -m * c, 1
        A[5, 3], A[5, 5] = m * l, -m * l
        # AH
        A[6, 4], A[6, 5], A[6, 6], A[6, 7] = c, l, -c, -l
        A[7, 6], A[7, 7] = psy.wsp(θs0), -1
        b[7] = psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # MX2
        A[8, 4], A[8, 6], A[8, 8] = β * m * c, (1 - β) * m * c, -m * c
        A[9, 5], A[9, 7], A[9, 9] = β * m * l, (1 - β) * m * l, -m * l
        # HC2
        A[10, 8], A[10, 10], A[10, 19] = m * c, -m * c, 1
        A[11, 9], A[11, 11] = m * l, -m * l
        # TZ
        A[12, 10], A[12, 12], A[12, 20] = m * c, -m * c, 1
        A[13, 11], A[13, 13], A[13, 21] = m * l, -m * l, 1
        # BL
        A[14, 12], A[14, 20], b[14] = (UA + mi * c), 1, (UA + mi * c) * θO + Qsa
        A[15, 13], A[15, 21], b[15] = mi * l, 1, mi * l * wO + Qla
        # Kθ & Kw
        A[16, 12], A[16, 18], b[16] = Kθ, 1, Kθ * θIsp
        A[17, 13], A[17, 19], b[17] = Kw, 1, Kw * wIsp
        # XC
        A[18, 12], A[18, 14], A[18, 23] = (1 - β_HX) * m * c, -(1 - β_HX) * m * c, -1
        A[19, 13], A[19, 15], A[19, 24] = (1 - β_HX) * m * l, -(1 - β_HX) * m * l, -1
        # XM
        A[20, 12], A[20, 14], A[20, 16] = β_HX, (1 - β_HX), -1
        A[21, 13], A[21, 15], A[21, 17] = β_HX, (1 - β_HX), -1
        # Q
        A[22, 0], A[22, 12], A[22, 14], A[22, 23], A[22, 24], b[22] = UA_HX, -UA_HX, -UA_HX, 2, 2, -UA_HX * θO
        A[23, 14], A[23, 15], b[23] = psy.wsp(θs1), -1, psy.wsp(θs1) * θs1 - psy.w(θs1, 1)
        A[24, 22], A[24, 23], A[24, 24] = 1, -1, -1

        x = np.linalg.solve(A, b)
        Δ_θs0 = abs(θs0 - x[6])
        θs0 = x[6]
        Δ_θs1 = abs(θs1 - x[14])
        θs1 = x[14]
    return x


def ModelRecAirmxmaHX(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX):
    """
    Model:
        Heat Exchanger
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
        β_HX    by-pass factor of the eat exchanger, -
        θS      supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp    indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO      outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latente heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K
        UA_HX   global conductivity HX, W/K
    System:
        XH:     Heating in Heat Exchanger
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
        XC:     Cooling in Heat Exchanger
        XM:     Mixing in Heat Exchanger
        Qs:     Exchanged sensible Heat
        Ql:     Exchanged latent Heat
        o:      outdoor conditions
        0..9    unknown points (temperature, humidity ratio)

        |--------|
    <-9-XM       |<---|<-----------------------------------------------------|
        |        |    |                                                      |
        |<-8-XC--|    |                                                      |
            |  |      |                                                      |
            Qs+Ql     |                                                      |
             |        /             |-------|                                |
        --o->XH--0->MX1--1->HC1--2->|       MX2--4->MX_AD2--5->HC2--6->TZ--7-|
                            /       |       |                   /      ||    |
                            |       |->AH-3-|                   |      BL    |
                            |                                   |            |
                            |                                   |<-----Kθ----|<-t7
                            |<-----------------------------------------Kw----|<-w7

    Returns
    -------
    x       vector 27 elem.:
            θ0, w0, θ1, w1, θ2, w2, θ3, w3, θ4, w4, θ5, w5, θ6, w6, θ7, w7, θ8, w8, θ9, w9,
            QHC1, QHC2, QsTZ, QlTZ, Qx, Qs, Ql
    """

    Kθ, Kw = 1e10, 1e10  # controller gain
    wO = psy.w(θO, φO)  # hum. out
    wIsp = psy.w(θIsp, φIsp)  # indoor humidity ratio

    # Model
    θs0, Δ_θs0 = θS, 2  # initial guess saturation temp.
    θs1, Δ_θs1 = θS, 2  # initial guess saturation temp.
    θs2, Δ_θs2 = θS, 2  # initial guess saturation temp.

    A = np.zeros((27, 27))  # coefficients of unknowns
    b = np.zeros(27)  # vector of inputs
    while Δ_θs0 > 0.01 or Δ_θs1 > 0.01 or Δ_θs2 > 0.01:
        # XH
        A[0, 0], A[0, 24], b[0] = -m * c, 1, -m * c * θO
        A[1, 1], b[1] = 1, wO
        # MX1
        A[2, 0], A[2, 2], A[2, 14] = α * m * c, - m * c, (1 - α) * m * c
        A[3, 1], A[3, 3], A[3, 15] = α * m * l, -m * l, (1 - α) * m * l
        # HC1
        A[4, 2], A[4, 4], A[4, 20] = m * c, -m * c, 1
        A[5, 3], A[5, 5] = m * l, -m * l
        # AH
        A[6, 4], A[6, 5], A[6, 6], A[6, 7] = c, l, -c, -l
        A[7, 6], A[7, 7] = psy.wsp(θs0), -1
        b[7] = psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # MX2
        A[8, 4], A[8, 6], A[8, 8] = β * m * c, (1 - β) * m * c, -m * c
        A[9, 5], A[9, 7], A[9, 9] = β * m * l, (1 - β) * m * l, -m * l
        # MX_AD2
        A[10, 8], A[10, 9], A[10, 10], A[10, 11] = c, l, -c, -l
        A[11, 10], A[11, 11], b[11] = psy.wsp(θs1), -1, psy.wsp(θs1) * θs1 - psy.w(θs1, 1)
        # HC2
        A[12, 10], A[12, 12], A[12, 21] = m * c, -m * c, 1
        A[13, 11], A[13, 13] = m * l, -m * l
        # TZ
        A[14, 12], A[14, 14], A[14, 22] = m * c, -m * c, 1
        A[15, 13], A[15, 15], A[15, 23] = m * l, -m * l, 1
        # BL
        A[16, 14], A[16, 22], b[16] = (UA + mi * c), 1, (UA + mi * c) * θO + Qsa
        A[17, 15], A[17, 23], b[17] = mi * l, 1, mi * l * wO + Qla
        # Kθ & Kw
        A[18, 14], A[18, 20], b[18] = Kθ, 1, Kθ * θIsp
        A[19, 15], A[19, 21], b[19] = Kw, 1, Kw * wIsp
        # XC
        A[20, 14], A[20, 16], A[20, 25] = (1 - β_HX) * m * c, -(1 - β_HX) * m * c, -1
        A[21, 15], A[21, 17], A[21, 26] = (1 - β_HX) * m * l, -(1 - β_HX) * m * l, -1
        # XM
        A[22, 14], A[22, 16], A[22, 18] = β_HX, (1 - β_HX), -1
        A[23, 15], A[23, 19], A[23, 19] = β_HX, (1 - β_HX), -1
        # Q
        A[24, 0], A[24, 14], A[24, 16], A[24, 25], A[24, 26], b[24] = UA_HX, -UA_HX, -UA_HX, 2, 2, -UA_HX * θO
        A[25, 16], A[25, 17], b[25] = psy.wsp(θs2), -1, psy.wsp(θs2) * θs2 - psy.w(θs2, 1)
        A[26, 24], A[26, 25], A[26, 26] = 1, -1, -1

        x = np.linalg.solve(A, b)
        Δ_θs0 = abs(θs0 - x[6])
        θs0 = x[6]
        Δ_θs1 = abs(θs1 - x[10])
        θs1 = x[10]
        Δ_θs2 = abs(θs2 - x[16])
        θs2 = x[16]
    return x


def ModelRecAirmamxHX(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX):
    """
    Model:
        Heat Exchanger
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
        β_HX    by-pass factor of the eat exchanger, -
        θS      supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp    indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO      outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latente heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K
        UA_HX   global conductivity HX, W/K
    System:
        XH:     Heating in Heat Exchanger
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
        XC:     Cooling in Heat Exchanger
        XM:     Mixing in Heat Exchanger
        Qs:     Exchanged sensible Heat
        Ql:     Exchanged latent Heat
        o:      outdoor conditions
        0..9    unknown points (temperature, humidity ratio)

        |--------|
    <-9-XM       |<---|<-----------------------------------------------------|
        |        |    |                                                      |
        |<-8-XC--|    |                                                      |
            |  |      |                                                      |
            Qs+Ql     |                                                      |
             |        /                        |-------|                     |
        --o->XH--0->MX1--1->MX_AD1--2->HC1--3->|       MX2--5->HC2--6->TZ--7-|
                                       /       |       |        /      ||    |
                                       |       |->AH-4-|        |      BL    |
                                       |                        |            |
                                       |                        |<-----Kθ----|<-t7
                                       |<------------------------------Kw----|<-w7

    Returns
    -------
    x       vector 27 elem.:
            θ0, w0, θ1, w1, θ2, w2, θ3, w3, θ4, w4, θ5, w5, θ6, w6, θ7, w7, θ8, w8, θ9, w9,
            QHC1, QHC2, QsTZ, QlTZ, Qx, Qs, Ql
    """
    Kθ, Kw = 1e10, 1e10  # controller gain
    wO = psy.w(θO, φO)  # hum. out
    wIsp = psy.w(θIsp, φIsp)  # indoor humidity ratio

    # Model
    θs0, Δ_θs0 = θS, 2  # initial guess saturation temp.
    θs1, Δ_θs1 = θS, 2  # initial guess saturation temp.
    θs2, Δ_θs2 = θS, 2  # initial guess saturation temp.

    A = np.zeros((27, 27))  # coefficients of unknowns
    b = np.zeros(27)  # vector of inputs
    while Δ_θs0 > 0.01 or Δ_θs1 > 0.01 or Δ_θs2 > 0.01:
        # XH
        A[0, 0], A[0, 24], b[0] = -m * c, 1, -m * c * θO
        A[1, 1], b[1] = 1, wO
        # MX1
        A[2, 0], A[2, 2], A[2, 14] = α * m * c, -m * c, (1 - α) * m * c
        A[3, 1], A[3, 3], A[3, 15] = α * m * l, -m * l, (1 - α) * m * l
        # MX_AD1
        A[4, 2], A[4, 3], A[4, 4], A[4, 5] = c, l, -c, -l
        A[5, 4], A[5, 5], b[5] = psy.wsp(θs0), -1, psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # HC1
        A[6, 4], A[6, 6], A[6, 20] = m * c, -m * c, 1
        A[7, 5], A[7, 7] = m * l, -m * l
        # AH
        A[8, 6], A[8, 7], A[8, 8], A[8, 9] = c, l, -c, -l
        A[9, 8], A[9, 9] = psy.wsp(θs1), -1
        b[9] = psy.wsp(θs1) * θs1 - psy.w(θs1, 1)
        # MX2
        A[10, 6], A[10, 8], A[10, 10] = β * m * c, (1 - β) * m * c, -m * c
        A[11, 7], A[11, 9], A[11, 11] = β * m * l, (1 - β) * m * l, -m * l
        # HC2
        A[12, 10], A[12, 12], A[12, 21] = m * c, -m * c, 1
        A[13, 11], A[13, 13] = m * l, -m * l
        # TZ
        A[14, 12], A[14, 14], A[14, 22] = m * c, -m * c, 1
        A[15, 13], A[15, 15], A[15, 23] = m * l, -m * l, 1
        # BL
        A[16, 14], A[16, 22], b[16] = (UA + mi * c), 1, (UA + mi * c) * θO + Qsa
        A[17, 15], A[17, 23], b[17] = mi * l, 1, mi * l * wO + Qla
        # Kθ & Kw
        A[18, 14], A[18, 20], b[18] = Kθ, 1, Kθ * θIsp
        A[19, 15], A[19, 21], b[19] = Kw, 1, Kw * wIsp
        # XC
        A[20, 14], A[20, 16], A[20, 25] = (1 - β_HX) * m * c, -(1 - β_HX) * m * c, -1
        A[21, 15], A[21, 17], A[21, 26] = (1 - β_HX) * m * l, -(1 - β_HX) * m * l, -1
        # XM
        A[22, 14], A[22, 16], A[22, 18] = β_HX, (1 - β_HX), -1
        A[23, 15], A[23, 19], A[23, 19] = β_HX, (1 - β_HX), -1
        # Q
        A[24, 0], A[24, 14], A[24, 16], A[24, 25], A[24, 26], b[24] = UA_HX, -UA_HX, -UA_HX, 2, 2, -UA_HX * θO
        A[25, 16], A[25, 17], b[25] = psy.wsp(θs2), -1, psy.wsp(θs2) * θs2 - psy.w(θs2, 1)
        A[26, 24], A[26, 25], A[26, 26] = 1, -1, -1

        x = np.linalg.solve(A, b)
        Δ_θs0 = abs(θs0 - x[8])
        θs0 = x[8]
        Δ_θs1 = abs(θs1 - x[4])
        θs1 = x[4]
        Δ_θs2 = abs(θs2 - x[16])
        θs2 = x[16]
    return x


def ModelRecAirmamaHX(m, α, β, β_HX, θS, θIsp, φIsp, θO, φO, Qsa, Qla, mi, UA, UA_HX):
    """
    Model:
        Heat Exchanger
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
        β_HX    by-pass factor of the eat exchanger, -
        θS      supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp    indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO      outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latente heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K
        UA_HX   global conductivity HX, W/K
    System:
        XH:     Heating in Heat Exchanger
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
        XC:     Cooling in Heat Exchanger
        XM:     Mixing in Heat Exchanger
        Qs:     Exchanged sensible Heat
        Ql:     Exchanged latent Heat
        o:      outdoor conditions
        0..10   unknown points (temperature, humidity ratio)

         |--------|
    <-10-XM       |<---|<---------------------------------------------------------------|
         |        |    |                                                                |
         |<-9-XC--|    |                                                                |
            |  |       |                                                                |
            Qs+Ql      |                                                                |
             |        /                        |-------|                                |
        --o->XH--0->MX1--1->MX_AD1--2->HC1--3->|       MX2--5->MX_AD2--6->HC2--7->TZ--8-|
                                       /       |       |                   /      ||    |
                                       |       |->AH-4-|                   |      BL    |
                                       |                                   |            |
                                       |                                   |<-----Kθ----|<-t8
                                       |<-----------------------------------------Kw----|<-w8

    Returns
    -------
    x       vector 29 elem.:
            θ0, w0, θ1, w1, θ2, w2, θ3, w3, θ4, w4, θ5, w5, θ6, w6, θ7, w7, θ8, w8, θ9, w9, θ10, w10,
            QHC1, QHC2, QsTZ, QlTZ, Qx, Qs, Ql
    """

    Kθ, Kw = 1e10, 1e10  # controller gain
    wO = psy.w(θO, φO)  # hum. out
    wIsp = psy.w(θIsp, φIsp)  # indoor humidity ratio

    # Model
    θs0, Δ_θs0 = θS, 2  # initial guess saturation temp.
    θs1, Δ_θs1 = θS, 2  # initial guess saturation temp.
    θs2, Δ_θs2 = θS, 2  # initial guess saturation temp.
    θs3, Δ_θs3 = θS, 2  # initial guess saturation temp.

    A = np.zeros((29, 29))  # coefficients of unknowns
    b = np.zeros(29)  # vector of inputs
    while Δ_θs0 > 0.01 or Δ_θs1 > 0.01 or Δ_θs2 > 0.01 or Δ_θs3 > 0.01:
        # XH
        A[0, 0], A[0, 26], b[0] = -m * c, 1, -m * c * θO
        A[1, 1], b[1] = 1, wO
        # MX1
        A[2, 0], A[2, 2], A[2, 12] = α * m * c, -m * c, (1 - α) * m * c
        A[3, 1], A[3, 3], A[3, 13] = α * m * l, -m * l, (1 - α) * m * l
        # MX_AD1
        A[4, 2], A[4, 3], A[4, 4], A[4, 5] = c, l, -c, -l
        A[5, 4], A[5, 5], b[5] = psy.wsp(θs0), -1, psy.wsp(θs0) * θs0 - psy.w(θs0, 1)
        # HC1
        A[6, 4], A[6, 6], A[6, 22] = m * c, -m * c, 1
        A[7, 5], A[7, 7] = m * l, -m * l
        # AH
        A[8, 6], A[8, 7], A[8, 8], A[8, 9] = c, l, -c, -l
        A[9, 8], A[9, 9] = psy.wsp(θs1), -1
        b[9] = psy.wsp(θs1) * θs1 - psy.w(θs1, 1)
        # MX2
        A[10, 6], A[10, 8], A[10, 10] = β * m * c, (1 - β) * m * c, -m * c
        A[11, 7], A[11, 9], A[11, 11] = β * m * l, (1 - β) * m * l, -m * l
        # MX_AD2
        A[12, 10], A[12, 11], A[12, 12], A[12, 13] = c, l, -c, -l
        A[13, 12], A[13, 13], b[13] = psy.wsp(θs2), -1, psy.wsp(θs2) * θs2 - psy.w(θs2, 1)
        # HC2
        A[14, 12], A[14, 14], A[14, 23] = m * c, -m * c, 1
        A[15, 13], A[15, 15] = m * l, -m * l
        # TZ
        A[16, 14], A[16, 16], A[16, 24] = m * c, -m * c, 1
        A[17, 15], A[17, 17], A[17, 25] = m * l, -m * l, 1
        # BL
        A[18, 16], A[18, 24], b[18] = (UA + mi * c), 1, (UA + mi * c) * θO + Qsa
        A[19, 17], A[19, 25], b[19] = mi * l, 1, mi * l * wO + Qla
        # Kθ & Kw
        A[20, 16], A[20, 22], b[20] = Kθ, 1, Kθ * θIsp
        A[21, 17], A[21, 23], b[21] = Kw, 1, Kw * wIsp
        # XC
        A[22, 16], A[22, 18], A[22, 27] = (1 - β_HX) * m * c, -(1 - β_HX) * m * c, -1
        A[23, 17], A[23, 19], A[23, 28] = (1 - β_HX) * m * l, -(1 - β_HX) * m * l, -1
        # XM
        A[24, 16], A[24, 18], A[24, 20] = β_HX, (1 - β_HX), -1
        A[25, 17], A[25, 19], A[25, 21] = β_HX, (1 - β_HX), -1
        # Q
        A[26, 0], A[26, 16], A[26, 18], A[26, 27], A[26, 28], b[26] = UA_HX, -UA_HX, -UA_HX, 2, 2, -UA_HX * θO
        A[27, 18], A[27, 19], b[27] = psy.wsp(θs3), -1, psy.wsp(θs3) * θs3 - psy.w(θs3, 1)
        A[28, 26], A[28, 27], A[28, 28] = 1, -1, -1

        x = np.linalg.solve(A, b)
        Δ_θs0 = abs(θs0 - x[8])
        θs0 = x[8]
        Δ_θs1 = abs(θs1 - x[4])
        θs1 = x[4]
        Δ_θs2 = abs(θs2 - x[12])
        θs2 = x[12]
        Δ_θs3 = abs(θs3 - x[18])
        θs3 = x[18]
    return x


def RecAirVAVmxmxHX(m=3, α=1, β=0.1, β_HX=0.1,
                    θSsp=30, θIsp=18, φIsp=0.49, θO=-1, φO=1,
                    Qsa=0, Qla=0, mi=2.18, UA=935.83, UA_HX=5000, check=True):
    """
    Heat Exchanger & Heating & Adiabatic humidification & Re-heating
    Recirculated air
    VAV Variable Air Volume:
        mass flow rate calculated to have const. supply temp.

    INPUTS:
        α       mixing ratio of outdoor air, -
        β       by-pass factor of the adiabatic humidifier, -
        β_HX    by-pass factor of the eat exchanger, -
        θSsp    supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp    indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO      outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latent heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K
        UA_HX   global conductivity HX, W/K
    System:
        XH:     Heating in Heat Exchanger
        MX1:    Mixing box
        HC1:    Heating Coil
        AH:     Adiabatic Humidifier
        MX2:    Mixing in humidifier model
        HC2:    Reheating coil
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kθ:     Controller - temperature
        XC:     Cooling in Heat Exchanger
        XM:     Mixing in Heat Exchanger
        Qs:     Sensible Heat
        Ql:     Latent Heat
        o:      outdoor conditions
        0..8    unknown points (temperature, humidity ratio)

        |--------|
    <-8-XM       |<---|<------------------------------------------|
        |        |    |                                           |
        |<-7-XC--|    |                                           |
            |  |      |                                           |
            Qs+Ql     |                                           |
             |        /             |-------|                     |
        --o->XH--0->MX1--1->HC1--2->|       MX2--4->HC2--5->TZ--6-|
                            /       |       |        /      ||    |
                            |       |->AH-3-|        |      BL    |
                            |                        |            |
                            |                        |<-----Kθ----|<-t6
                            |<------------------------------Kw----|<-w6

    25 Unknowns
        0..8: 2*9 points (temperature, humidity ratio)
        QsHC1, QsHC2, QsTZ, QlTZ, Qx, Qs, Ql
    """

    plt.close('all')
    wO = psy.w(θO, φO)  # hum. out

    x = ModelRecAirmxmxHX(m, α, β, β_HX, θSsp, θIsp, φIsp,
                          θO, φO, Qsa, Qla, mi, UA, UA_HX)

    if not check:
        θ = np.append(θO, x[0:17:2])
        w = np.append(wO, x[1:18:2])

        # Adjacency matrix
        # Points calc.  o   0   1   2   3   4   5   6   7   8       Elements
        # Points plot   0   1   2   3   4   5   6   7   8   9       Elements
        A = np.array([[-1, +1, +0, +0, +0, +0, +0, +0, +0, +0],  # XH
                      [+0, -1, +1, +0, +0, +0, +0, -1, +0, +0],  # MX1
                      [+0, +0, -1, +1, +0, +0, +0, +0, +0, +0],  # HC1
                      [+0, +0, +0, -1, +1, +0, +0, +0, +0, +0],  # AH
                      [+0, +0, +0, -1, -1, +1, +0, +0, +0, +0],  # MX2
                      [+0, +0, +0, +0, +0, -1, +1, +0, +0, +0],  # HC2
                      [+0, +0, +0, +0, +0, +0, -1, +1, +0, +0],  # TZ
                      [+0, +0, +0, +0, +0, +0, +0, -1, +1, +0],  # XC
                      [+0, +0, +0, +0, +0, +0, +0, -1, -1, +1]])  # XM

        psy.chartA(θ, w, A)

        θ = pd.Series(θ)
        w = 1000 * pd.Series(w)
        P = pd.concat([θ, w], axis=1)  # points
        P.columns = ['θ [°C]', 'w [g/kg]']

        output = P.to_string(formatters={
            'θ [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format
        })
        print()
        print(output)

        Q = pd.Series(x[18:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ', 'Qx', 'Qs', 'Ql'])
        # Q.columns = ['kW']
        pd.options.display.float_format = '{:,.2f}'.format
        print()
        print(Q.to_frame().T / 1000, 'kW')

        return None
    else:
        return x


def RecAirVAVmxmaHX(m=3, α=1, β=0.1, β_HX=0.1,
                    θSsp=30, θIsp=18, φIsp=0.49, θO=-1, φO=1,
                    Qsa=0, Qla=0, mi=2.18, UA=935.83, UA_HX=5000, check=True):
    """
    Heat Exchanger & Heating & Adiabatic Mixing at second Mixer & Adiabatic humidification & Re-heating
    Recirculated air
    VAV Variable Air Volume:
        mass flow rate calculated to have const. supply temp.

    INPUTS:
        α       mixing ratio of outdoor air, -
        β       by-pass factor of the adiabatic humidifier, -
        β_HX    by-pass factor of the eat exchanger, -
        θS      supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp    indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO      outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latente heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K
        UA_HX   global conductivity HX, W/K
    System:
        XH:     Heating in Heat Exchanger
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
        XC:     Cooling in Heat Exchanger
        XM:     Mixing in Heat Exchanger
        Qs:     Exchanged sensible Heat
        Ql:     Exchanged latent Heat
        o:      outdoor conditions
        0..9    unknown points (temperature, humidity ratio)

        |--------|
    <-9-XM       |<---|<-----------------------------------------------------|
        |        |    |                                                      |
        |<-8-XC--|    |                                                      |
            |  |      |                                                      |
            Qs+Ql     |                                                      |
             |        /             |-------|                                |
        --o->XH--0->MX1--1->HC1--2->|       MX2--4->MX_AD2--5->HC2--6->TZ--7-|
                            /       |       |                   /      ||    |
                            |       |->AH-3-|                   |      BL    |
                            |                                   |            |
                            |                                   |<-----Kθ----|<-t7
                            |<-----------------------------------------Kw----|<-w7

    27 Unknowns
        0..9: 2*10 points (temperature, humidity ratio)
        QsHC1, QsHC2, QsTZ, QlTZ, Qx, Qs, Ql
    """
    plt.close('all')
    wO = psy.w(θO, φO)  # hum. out

    x = ModelRecAirmxmaHX(m, α, β, β_HX, θSsp, θIsp, φIsp,
                          θO, φO, Qsa, Qla, mi, UA, UA_HX)

    if not check:
        θ = np.append(θO, x[0:19:2])
        w = np.append(wO, x[1:20:2])

        # Adjacency matrix
        # Points calc.  o   0   1   2   3   4   5   6   7   8   9       Elements
        # Points plot   0   1   2   3   4   5   6   7   8   9   10      Elements
        A = np.array([[-1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +0],  # XH
                      [+0, -1, +1, +0, +0, +0, +0, +0, -1, +0, +0],  # MX1
                      [+0, +0, -1, +1, +0, +0, +0, +0, +0, +0, +0],  # HC1
                      [+0, +0, +0, -1, +1, +0, +0, +0, +0, +0, +0],  # AH
                      [+0, +0, +0, -1, -1, +1, +0, +0, +0, +0, +0],  # MX2
                      [+0, +0, +0, +0, +0, -1, +1, +0, +0, +0, +0],  # MX_AD2
                      [+0, +0, +0, +0, +0, +0, -1, +1, +0, +0, +0],  # HC2
                      [+0, +0, +0, +0, +0, +0, +0, -1, +1, +0, +0],  # TZ
                      [+0, +0, +0, +0, +0, +0, +0, +0, -1, +1, +0],  # XC
                      [+0, +0, +0, +0, +0, +0, +0, +0, -1, -1, +1]])  # XM

        psy.chartA(θ, w, A)

        θ = pd.Series(θ)
        w = 1000 * pd.Series(w)
        P = pd.concat([θ, w], axis=1)  # points
        P.columns = ['θ [°C]', 'w [g/kg]']

        output = P.to_string(formatters={
            'θ [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format
        })
        print()
        print(output)

        Q = pd.Series(x[20:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ', 'Qx', 'Qs', 'Ql'])
        # Q.columns = ['kW']
        pd.options.display.float_format = '{:,.2f}'.format
        print()
        print(Q.to_frame().T / 1000, 'kW')

        return None
    else:
        return x


def RecAirVAVmamxHX(m=3, α=1, β=0.1, β_HX=0.1,
                    θSsp=30, θIsp=18, φIsp=0.49, θO=-1, φO=1,
                    Qsa=0, Qla=0, mi=2.18, UA=935.83, UA_HX=5000, check=True):
    """
    Heat Exchanger & Heating & Adiabatic Mixing at first Mixer & Adiabatic humidification & Re-heating
    Recirculated air
    VAV Variable Air Volume:
        mass flow rate calculated to have const. supply temp.

    INPUTS:
        α       mixing ratio of outdoor air, -
        β       by-pass factor of the adiabatic humidifier, -
        β_HX    by-pass factor of the eat exchanger, -
        θS      supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp    indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO      outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latente heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K
        UA_HX   global conductivity HX, W/K
    System:
        XH:     Heating in Heat Exchanger
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
        XC:     Cooling in Heat Exchanger
        XM:     Mixing in Heat Exchanger
        Qs:     Exchanged sensible Heat
        Ql:     Exchanged latent Heat
        o:      outdoor conditions
        0..9    unknown points (temperature, humidity ratio)

        |--------|
    <-9-XM       |<---|<-----------------------------------------------------|
        |        |    |                                                      |
        |<-8-XC--|    |                                                      |
            |  |      |                                                      |
            Qs+Ql     |                                                      |
             |        /                        |-------|                     |
        --o->XH--0->MX1--1->MX_AD1--2->HC1--3->|       MX2--5->HC2--6->TZ--7-|
                                       /       |       |        /      ||    |
                                       |       |->AH-4-|        |      BL    |
                                       |                        |            |
                                       |                        |<-----Kθ----|<-t7
                                       |<------------------------------Kw----|<-w7

    27 Unknowns
        0..9: 2*10 points (temperature, humidity ratio)
        QsHC1, QsHC2, QsTZ, QlTZ, Qx, Qs, Ql
    """
    plt.close('all')
    wO = psy.w(θO, φO)  # hum. out

    x = ModelRecAirmamxHX(m, α, β, β_HX, θSsp, θIsp, φIsp,
                          θO, φO, Qsa, Qla, mi, UA, UA_HX)

    if not check:
        θ = np.append(θO, x[0:19:2])
        w = np.append(wO, x[1:20:2])

        # Adjacency matrix
        # Points calc.  o   0   1   2   3   4   5   6   7   8   9       Elements
        # Points plot   0   1   2   3   4   5   6   7   8   9   10      Elements
        A = np.array([[-1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +0],  # XH
                      [+0, -1, +1, +0, +0, +0, +0, +0, -1, +0, +0],  # MX1
                      [+0, +0, -1, +1, +0, +0, +0, +0, +0, +0, +0],  # MX_AD1
                      [+0, +0, +0, -1, +1, +0, +0, +0, +0, +0, +0],  # HC1
                      [+0, +0, +0, +0, -1, +1, +0, +0, +0, +0, +0],  # AH
                      [+0, +0, +0, +0, -1, -1, +1, +0, +0, +0, +0],  # MX2
                      [+0, +0, +0, +0, +0, +0, -1, +1, +0, +0, +0],  # HC2
                      [+0, +0, +0, +0, +0, +0, +0, -1, +1, +0, +0],  # TZ
                      [+0, +0, +0, +0, +0, +0, +0, +0, -1, +1, +0],  # XC
                      [+0, +0, +0, +0, +0, +0, +0, +0, -1, -1, +1]])  # XM

        psy.chartA(θ, w, A)

        θ = pd.Series(θ)
        w = 1000 * pd.Series(w)
        P = pd.concat([θ, w], axis=1)  # points
        P.columns = ['θ [°C]', 'w [g/kg]']

        output = P.to_string(formatters={
            'θ [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format
        })
        print()
        print(output)

        Q = pd.Series(x[20:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ', 'Qx', 'Qs', 'Ql'])
        # Q.columns = ['kW']
        pd.options.display.float_format = '{:,.2f}'.format
        print()
        print(Q.to_frame().T / 1000, 'kW')

        return None
    else:
        return x

def RecAirVAVmamaHX(m=3, α=1, β=0.1, β_HX=0.1,
                    θSsp=30, θIsp=18, φIsp=0.49, θO=-1, φO=1,
                    Qsa=0, Qla=0, mi=2.18, UA=935.83, UA_HX=5000, check=True):
    """
    Heat Exchanger & Heating & Adiabatic Mixing at both Mixers & Adiabatic humidification & Re-heating
    Recirculated air
    VAV Variable Air Volume:
        mass flow rate calculated to have const. supply temp.

    INPUTS:
        α       mixing ratio of outdoor air, -
        β       by-pass factor of the adiabatic humidifier, -
        β_HX    by-pass factor of the eat exchanger, -
        θS      supply air, °C
        θIsp    indoor air setpoint, °C
        φIsp    indoor relative humidity set point, -
        θO      outdoor temperature for design, °C
        φO      outdoor relative humidity for design, -
        Qsa     aux. sensible heat, W
        Qla     aux. latente heat, W
        mi      infiltration massflow rate, kg/s
        UA      global conductivity bldg, W/K
        UA_HX   global conductivity HX, W/K
    System:
        XH:     Heating in Heat Exchanger
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
        XC:     Cooling in Heat Exchanger
        XM:     Mixing in Heat Exchanger
        Qs:     Exchanged sensible Heat
        Ql:     Exchanged latent Heat
        o:      outdoor conditions
        0..10    unknown points (temperature, humidity ratio)

         |--------|
    <-10-XM       |<---|<---------------------------------------------------------------|
         |        |    |                                                                |
         |<-9-XC--|    |                                                                |
            |  |       |                                                                |
            Qs+Ql      |                                                                |
             |        /                        |-------|                                |
        --o->XH--0->MX1--1->MX_AD1--2->HC1--3->|       MX2--5->MX_AD2--6->HC2--7->TZ--8-|
                                       /       |       |                   /      ||    |
                                       |       |->AH-4-|                   |      BL    |
                                       |                                   |            |
                                       |                                   |<-----Kθ----|<-t8
                                       |<-----------------------------------------Kw----|<-w8

    29 Unknowns
        0..10: 2*11 points (temperature, humidity ratio)
        QsHC1, QsHC2, QsTZ, QlTZ, Qx, Qs, Ql
    """
    plt.close('all')
    wO = psy.w(θO, φO)  # hum. out

    x = ModelRecAirmamaHX(m, α, β, β_HX, θSsp, θIsp, φIsp,
                          θO, φO, Qsa, Qla, mi, UA, UA_HX)

    if not check:
        θ = np.append(θO, x[0:20:2])
        w = np.append(wO, x[1:21:2])

        # Adjacency matrix
        # Points calc.  o   0   1   2   3   4   5   6   7   8   9   10      Elements
        # Points plot   0   1   2   3   4   5   6   7   8   9   10  11      Elements
        A = np.array([[-1, +1, +0, +0, +0, +0, +0, +0, +0, +0, +0, +0],  # XH
                      [+0, -1, +1, +0, +0, +0, +0, +0, +0, -1, +0, +0],  # MX1
                      [+0, +0, -1, +1, +0, +0, +0, +0, +0, +0, +0, +0],  # MX_AD1
                      [+0, +0, +0, -1, +1, +0, +0, +0, +0, +0, +0, +0],  # HC1
                      [+0, +0, +0, +0, -1, +1, +0, +0, +0, +0, +0, +0],  # AH
                      [+0, +0, +0, +0, -1, -1, +1, +0, +0, +0, +0, +0],  # MX2
                      [+0, +0, +0, +0, +0, +0, -1, +1, +0, +0, +0, +0],  # MX_AD2
                      [+0, +0, +0, +0, +0, +0, +0, -1, +1, +0, +0, +0],  # HC2
                      [+0, +0, +0, +0, +0, +0, +0, +0, -1, +1, +0, +0],  # TZ
                      [+0, +0, +0, +0, +0, +0, +0, +0, +0, -1, +1, +0],  # XC
                      [+0, +0, +0, +0, +0, +0, +0, +0, +0, -1, -1, +1]])  # XM

        psy.chartA(θ, w, A)

        θ = pd.Series(θ)
        w = 1000 * pd.Series(w)
        P = pd.concat([θ, w], axis=1)  # points
        P.columns = ['θ [°C]', 'w [g/kg]']

        output = P.to_string(formatters={
            'θ [°C]': '{:,.2f}'.format,
            'w [g/kg]': '{:,.2f}'.format
        })
        print()
        print(output)

        Q = pd.Series(x[22:], index=['QsHC1', 'QsHC2', 'QsTZ', 'QlTZ', 'Qx', 'Qs', 'Ql'])
        # Q.columns = ['kW']
        pd.options.display.float_format = '{:,.2f}'.format
        print()
        print(Q.to_frame().T / 1000, 'kW')

        return None
    else:
        return x