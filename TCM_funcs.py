"""
Created on We Nov 10 2021

@author: L. Beber, C. Gerike-Roberts, E. Regev

File with all the functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem
import math


def building_characteristics():
    """
    This code is designed to read an excel file which contains the characteristics of the building
    and create a data frame from it.
    """

    bc = pd.read_csv(r'Building Characteristics_real.csv', na_values=["N"], keep_default_na=True)

    return bc


def thphprop(BCdf):
    """
    Parameters
    ----------
    BCdf : data frame of building characteristics
        DESCRIPTION.
        Data Frame of building characteristics. Example:
                BCdf = ['Element Code', 'Element Type', 'Material 1', 'Material 2', 'Material 3', 'Length', 'Width',
                'Height', 'Thickness 1', 'Thickness 2', Thickness 3', 'Surface', 'Volume', 'Slope', 'Azimuth', ]

    Returns
    -------
    Bdf : data frame
        DESCRIPTION.
        data frame of the Building characteristics with associated thermophysical properties
                Bdf = ['Element Code', 'Element Type', 'Material 1', 'Material 2', 'Material 3', 'Length', 'Width',
                'Height', 'Thickness 1', 'Thickness 2', Thickness 3', 'Surface', 'Volume', 'Slope', 'Azimuth',
                'Density 1', 'specific heat 1', 'conductivity 1', 'LW emissivity 1', 'SW transmittance 1',
                'SW absorptivity 1', 'albedo 1', 'Density 2', 'specific heat 2', 'conductivity 2', 'LW emissivity 2',
                'SW transmittance 2', 'SW absorptivity 2', 'albedo 2', 'Density 3', 'specific heat 3', 'conductivity 3',
                'LW emissivity 3', 'SW transmittance 3', 'SW absorptivity 3', 'albedo 3']
    """

    # Thermo-physical and radiative properties - source data frame
    # ----------------------------------------------------------

    """ Incropera et al. (2011) Fundamentals of heat and mass transfer, 7 ed,
        Table A3,
            concrete (stone mix) p. 993
            insulation polystyrene extruded (R-12) p.990
            glass plate p.993
            Clay tile, hollow p.989
            Wood, oak p.989
            Soil p.994

        EngToolbox Emissivity Coefficient Materials, Glass, pyrex
        EngToolbox Emissivity Coefficient Materials, Clay
        EngToolbox Emissivity Coefficient Materials, Wood Oak, planned
        EngToolbox Absorbed Solar Radiation by Surface Color, white smooth surface
        EngToolbox Optical properties of some typical glazing mat Window glass
        EngToolbox Absorbed Solar Radiation by Material, Tile, clay red
        EngToolbox Absorbed Solar Radiation by Surface Color, Green, red and brown
        """
    thphp = {'Material': ['Concrete', 'Insulation', 'Glass', 'Air', 'Tile', 'Wood', 'Soil'],
             'Density': [2300, 55, 2500, 1.2, None, 720, 2050],  # kg/m³
             'Specific_Heat': [880, 1210, 750, 1000, None, 1255, 1840],  # J/kg.K
             'Conductivity': [1.4, 0.027, 1.4, None, 0.52, 0.16, 0.52],  # W/m.K
             'LW_Emissivity': [0.9, 0, 0.9, 0, 0.91, 0.885, None],
             'SW_Transmittance': [0, 0, 0.83, 1, 0, 0, 0],
             'SW_Absorptivity': [0.25, 0.25, 0.1, 0, 0.64, 0.6, None],
             'Albedo': [0.75, 0.75, 0.07, 0, 0.36, 0.4, None]}  # albedo + SW transmission + SW absorptivity = 1

    thphp = pd.DataFrame(thphp)

    # add empty columns for thermo-physical properties
    BCdf = BCdf.reindex(columns=BCdf.columns.to_list() + ['rad_s', 'density_1', 'specific_heat_1', 'conductivity_1',
                                                          'LW_emissivity_1', 'SW_transmittance_1', 'SW_absorptivity_1',
                                                          'albedo_1', 'density_2', 'specific_heat_2', 'conductivity_2',
                                                          'LW_emissivity_2', 'SW_transmittance_2', 'SW_absorptivity_2',
                                                          'albedo_2', 'density_3', 'specific_heat_3', 'conductivity_3',
                                                          'LW_emissivity_3', 'SW_transmittance_3', 'SW_absorptivity_3',
                                                          'albedo_3'])

    # fill columns with properties for the given materials 1-3 of each element
    for i in range(0, len(BCdf)):
        for j in range(0, len(thphp['Material'])):
            if BCdf.loc[i, 'Material_1'] == thphp.Material[j]:
                BCdf.loc[i, 'density_1'] = thphp.Density[j]
                BCdf.loc[i, 'specific_heat_1'] = thphp.Specific_Heat[j]
                BCdf.loc[i, 'conductivity_1'] = thphp.Conductivity[j]
                BCdf.loc[i, 'LW_emissivity_1'] = thphp.LW_Emissivity[j]
                BCdf.loc[i, 'SW_transmittance_1'] = thphp.SW_Transmittance[j]
                BCdf.loc[i, 'SW_absorptivity_1'] = thphp.SW_Absorptivity[j]
                BCdf.loc[i, 'albedo_1'] = thphp.Albedo[j]

        for j in range(0, len(thphp['Material'])):
            if BCdf.loc[i, 'Material_2'] == thphp.Material[j]:
                BCdf.loc[i, 'density_2'] = thphp.Density[j]
                BCdf.loc[i, 'specific_heat_2'] = thphp.Specific_Heat[j]
                BCdf.loc[i, 'conductivity_2'] = thphp.Conductivity[j]
                BCdf.loc[i, 'LW_emissivity_2'] = thphp.LW_Emissivity[j]
                BCdf.loc[i, 'SW_transmittance_2'] = thphp.SW_Transmittance[j]
                BCdf.loc[i, 'SW_absorptivity_2'] = thphp.SW_Absorptivity[j]
                BCdf.loc[i, 'albedo_2'] = thphp.Albedo[j]

        for j in range(0, len(thphp['Material'])):
            if BCdf.loc[i, 'Material_3'] == thphp.Material[j]:
                BCdf.loc[i, 'density_3'] = thphp.Density[j]
                BCdf.loc[i, 'specific_heat_3'] = thphp.Specific_Heat[j]
                BCdf.loc[i, 'conductivity_3'] = thphp.Conductivity[j]
                BCdf.loc[i, 'LW_emissivity_3'] = thphp.LW_Emissivity[j]
                BCdf.loc[i, 'SW_transmittance_3'] = thphp.SW_Transmittance[j]
                BCdf.loc[i, 'SW_absorptivity_3'] = thphp.SW_Absorptivity[j]
                BCdf.loc[i, 'albedo_3'] = thphp.Albedo[j]

    return BCdf


def rad(bcp, albedo_sur, latitude, dt, WF, t_start, t_end):
    # Simulation with weather data
    # ----------------------------
    filename = WF
    start_date = t_start
    end_date = t_end

    # Read weather data from Energyplus .epw file
    [data, meta] = dm4bem.read_epw(filename, coerce_year=None)
    weather = data[["temp_air", "relative_humidity", "dir_n_rad", "dif_h_rad", "atmospheric_pressure"]]
    del data
    weather.index = weather.index.map(lambda t: t.replace(year=2022))
    weather = weather[(weather.index >= start_date) & (
            weather.index < end_date)]
    # Solar radiation on a tilted surface South
    Φt = {}
    for k in range(0, len(bcp)):
        surface_orientationS = {'slope': bcp.loc[k, 'Slope'],
                                'azimuth': bcp.loc[k, 'Azimuth'],
                                'latitude': latitude}
        rad_surf = dm4bem.sol_rad_tilt_surf(weather, surface_orientationS, albedo_sur)
        Φt.update({str(k + 2): rad_surf.sum(axis=1)})

    Φt = pd.DataFrame(Φt)
    # Interpolate weather data for time step dt
    data = pd.concat([weather['temp_air'], weather['relative_humidity'], weather['atmospheric_pressure'], Φt], axis=1)
    data = data.resample(str(dt) + 'S').interpolate(method='linear')
    data = data.rename(columns={'temp_air': 'To'})
    data = data.rename(columns={'atmospheric_pressure': 'Pamb'})

    # time
    t = dt * np.arange(data.shape[0])

    return data, t


def indoor_air(bcp_sur, h, V, Qa, rad_surf_tot):
    """
    Input:
    bcp_sur, surface column of bcp dataframe
    h, convection dataframe
    V, Volume of the room (from bcp)
    Output: TCd, a dictionary of the all the matrices of the thermal circuit of the inside air
    """
    nt = len(bcp_sur) + 1
    nq = len(bcp_sur)

    nq_ones = np.ones(nq)
    A = np.diag(-nq_ones)
    A = np.c_[nq_ones, A]

    G = np.zeros(nq)
    for i in range(0, len(G)):
        G[i] = h['in'] * bcp_sur[i]*1.2
    G = np.diag(G)
    b = np.zeros(nq)
    C = np.zeros(nt)
    C[0] = (1.2 * 1000 * V) / 2  # Capacity air = Density*specific heat*V
    C = np.diag(C)
    f = np.zeros(nt)
    f[0] = 1
    y = np.zeros(nt)
    y[0] = 1
    Q = np.zeros((rad_surf_tot.shape[0], nt))
    Q[:, 0] = Qa
    Q[:, 1:nt] = 'NaN'
    T = np.zeros((rad_surf_tot.shape[0], nq))
    T[:, 0:nq] = 'NaN'

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return TCd


def ventilation(V, V_dot, Kpf, T_set, rad_surf_tot):
    """
    Input:
    V, Volume of the room (from bcp)
    V_dot
    Kp
    Output:
    TCd, a dictionary of the all the matrices describing the thermal circuit of the ventilation
    """
    Gv = V_dot * 1.2 * 1000  # Va_dot * air['Density'] * air['Specific heat']
    A = np.array([[1],
                  [1]])
    G = np.diag(np.hstack([Gv, Kpf]))
    b = np.array([1, 1])
    C = np.array([(1.2 * 1000 * V) / 2])
    f = 0
    y = 1
    Q = np.zeros((rad_surf_tot.shape[0], 1))
    Q[:, 0] = 'NaN'
    T = np.zeros((rad_surf_tot.shape[0], 2))
    T[:, 0] = rad_surf_tot['To']
    T[:, 1] = T_set['heating']

    vent_c = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return vent_c


def solid_wall_w_ins(bcp_r, h, rad_surf_tot, uc):
    """Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a solid wall with insulation
    """
    # Thermal conductances
    # Conduction
    G_cd_cm = bcp_r['conductivity_1'] / bcp_r['Thickness_1'] * bcp_r['Surface']  # concrete
    G_cd_in = bcp_r['conductivity_2'] / bcp_r['Thickness_2'] * bcp_r['Surface']  # insulation

    # Convection
    Gw = h * bcp_r['Surface']  # wall

    # Thermal capacities
    Capacity_cm = bcp_r['density_1'] * bcp_r['specific_heat_1'] * bcp_r['Surface'] * bcp_r['Thickness_1']
    Capacity_in = bcp_r['density_2'] * bcp_r['specific_heat_2'] * bcp_r['Surface'] * bcp_r['Thickness_2']

    # Thermal network
    # ---------------
    nq = 1 + 2 * (int(bcp_r['Mesh_1']) + int(bcp_r['Mesh_2']))
    nt = 1 + 2 * (int(bcp_r['Mesh_1']) + int(bcp_r['Mesh_2']))

    A = np.eye(nq + 1, nt)
    A = -np.diff(A, 1, 0).T

    nc = int(bcp_r['Mesh_1'])
    ni = int(bcp_r['Mesh_2'])
    Gcm = 2 * nc * [G_cd_cm]
    Gcm = 2 * nc * np.array(Gcm)
    Gim = 2 * ni * [G_cd_in]
    Gim = 2 * ni * np.array(Gim)
    G = np.diag(np.hstack([Gw['out'], Gcm, Gim]))

    b = np.zeros(nq)
    b[0] = 1

    Ccm = Capacity_cm / nc * np.mod(range(0, 2 * nc), 2)
    Cim = Capacity_in / ni * np.mod(range(0, 2 * ni), 2)
    C = np.diag(np.hstack([Ccm, Cim, 0]))

    f = np.zeros(nt)
    f[0] = f[-1] = 1

    y = np.zeros(nt)

    Q = np.zeros((rad_surf_tot.shape[0], nt))
    Q[:, 0] = bcp_r['SW_absorptivity_1'] * bcp_r['Surface'] * rad_surf_tot[str(uc)]
    Q[:, (nt - 1)] = -1
    uca = uc + 1
    Q[:, 1:(nt - 1)] = 'NaN'

    T = np.zeros((rad_surf_tot.shape[0], nq))
    T[:, 0] = rad_surf_tot['To']
    T[:, 1:nq] = 'NaN'

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return TCd, uca


def window(bcp_r, h, rad_surf_tot, uc):
    """Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a solid wall with insulation
    """
    nq = 2 * (int(bcp_r['Mesh_1']))
    nt = 2 * (int(bcp_r['Mesh_1']))

    A = np.array([[1, 0],
                  [-1, 1]])
    Ggo = h['out'] * bcp_r['Surface']
    Ggs = 1 / (1 / Ggo + 1 / (2 * bcp_r['conductivity_1']))
    G = np.diag(np.hstack([Ggs, 2 * bcp_r['conductivity_1']]))
    b = np.array([1, 0])
    C = np.diag([bcp_r['density_1'] * bcp_r['specific_heat_1'] * bcp_r['Surface'] * bcp_r['Thickness_1'], 0])
    f = np.array([1, 0])
    y = np.array([0, 0])

    Q = np.zeros((rad_surf_tot.shape[0], nt))
    IG_surface = bcp_r['Surface'] * rad_surf_tot[str(uc)]
    IGR = np.zeros([rad_surf_tot.shape[0], 1])
    IGR = IGR[:, 0] + (bcp_r['SW_transmittance_1'] * bcp_r['Surface'] * rad_surf_tot[str(uc)])
    IGR = np.array([IGR]).T
    Q[:, 0] = bcp_r['SW_absorptivity_1'] * IG_surface
    uca = uc + 1
    Q[:, 1:nt] = 'NaN'

    T = np.zeros((rad_surf_tot.shape[0], nq))
    T[:, 0] = rad_surf_tot['To']
    T[:, 1:nq] = 'NaN'

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return TCd, uca, IGR


def susp_floor(bcp_r, h, V, rad_surf_tot, uc, Tg):
    """Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    V, Volume of the room from bcp
    Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a suspended floor
    """
    nq = 1 + 2 * (int(bcp_r['Mesh_2']) + int(bcp_r['Mesh_3']))
    nt = 1 + 2 * (int(bcp_r['Mesh_2']) + int(bcp_r['Mesh_3']))

    A = np.array([[1, 0, 0, 0, 0],
                  [-1, 1, 0, 0, 0],
                  [0, -1, 1, 0, 0],
                  [0, 0, -1, 1, 0],
                  [0, 0, 0, -1, 1]])
    Gw = h * bcp_r['Surface']
    G_cd = bcp_r['conductivity_3'] / bcp_r['Thickness_3'] * bcp_r['Surface']  # wood
    G_cd_soil = bcp_r['conductivity_1'] / bcp_r['Thickness_1'] * bcp_r['Surface']
    G = np.diag(np.hstack(
        [G_cd_soil, Gw['in'], Gw['in'], G_cd, G_cd]))
    b = np.array([1, 0, 0, 0, 0])
    Capacity_w = bcp_r['density_3'] * bcp_r['specific_heat_3'] * bcp_r['Surface'] * bcp_r['Thickness_3']  # wood
    Capacity_a = bcp_r['density_2'] * bcp_r['specific_heat_2'] * V  # air
    C = np.diag([0, Capacity_a, 0, Capacity_w, 0])
    f = np.array([0, 0, 0, 0, 1])
    y = np.array([0, 0, 0, 0, 0])

    Q = np.zeros((rad_surf_tot.shape[0], nt))
    Q[:, 0] = bcp_r['SW_absorptivity_3'] * bcp_r['Surface'] * rad_surf_tot[str(uc)]
    Q[:, (nt - 1)] = -1
    uca = uc + 1
    Q[:, 0:(nt - 1)] = 'NaN'

    T = np.zeros((rad_surf_tot.shape[0], nq))
    T[:, 0] = Tg
    T[:, 1:nq] = 'NaN'

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return TCd, uca


def flat_roof_w_in(bcp_r, h, rad_surf_tot, uc):
    """Input:
    bcp_r, one row of the bcp dataframe
    h, convection dataframe
    Output: TCd, a dictionary of the all the matrices of one thermal circuit describing a flat roof with insulation
    """
    nq = 1 + 2 * (int(bcp_r['Mesh_1']))
    nt = 1 + 2 * (int(bcp_r['Mesh_1']))

    A = np.array([[-1, 0, 0],
                  [-1, 1, 0],
                  [0, -1, 1]])
    Gw = h * bcp_r['Surface']
    G_cd_in = bcp_r['conductivity_2'] / bcp_r['Thickness_2'] * bcp_r['Surface']  # insulation
    ni = int(bcp_r['Mesh_2'])
    Gim = 2 * ni * [G_cd_in]
    Gim = 2 * ni * np.array(Gim)
    G = np.diag(np.hstack([Gw['out'], Gim]))
    b = np.array([1, 0, 0])
    Capacity_i = bcp_r['density_2'] * bcp_r['specific_heat_2'] * bcp_r['Surface'] * bcp_r['Thickness_2']  # insulation
    C = np.diag([0, Capacity_i, 0])
    f = np.array([1, 0, 1])
    y = np.array([0, 0, 0])

    Q = np.zeros((rad_surf_tot.shape[0], nt))
    Q[:, 0] = bcp_r['SW_absorptivity_1'] * bcp_r['Surface'] * rad_surf_tot[str(uc)]
    Q[:, (nt - 1)] = -1
    uca = uc + 1
    Q[:, 1:(nt - 1)] = 'NaN'

    T = np.zeros((rad_surf_tot.shape[0], nq))
    T[:, 0] = rad_surf_tot['To']
    T[:, 1:nq] = 'NaN'

    TCd = {'A': A, 'G': G, 'b': b, 'C': C, 'f': f, 'y': y, 'Q': Q, 'T': T}

    return TCd, uca


def indoor_rad(bcp_r, TCd, IG):
    Q = TCd['Q']
    lim = np.shape(Q)[1]
    for i in range(0, lim):
        if Q[0, i] == -1:
            if np.isnan(bcp_r['SW_absorptivity_3']):
                if np.isnan(bcp_r['SW_absorptivity_2']):
                    x = bcp_r['SW_absorptivity_1'] * IG
                    Q[:, i] = x[:, 0]
                else:
                    x = bcp_r['SW_absorptivity_2'] * IG
                    Q[:, i] = x[:, 0]
            else:
                x = bcp_r['SW_absorptivity_3'] * IG
                Q[:, i] = x[:, 0]
    TCd['Q'] = Q  # replace Q in TCd with new Q

    return TCd

def indoor_rad_c(TCd_c):
    Q = TCd_c['Q']
    lim = np.shape(Q)[1]
    for i in range(0, lim):
        if Q[0, i] == -1:
            Q[:, i] = 0

    TCd_c['Q'] = Q  # replace Q in TCd with new Q

    return TCd_c


def u_assembly(TCd, rad_surf_tot):
    rad_surf_tot = rad_surf_tot.loc[:, rad_surf_tot.any()]
    u = np.empty((len(rad_surf_tot), 1))  # create u matrix
    for i in range(0, TCd.shape[1]):
        TCd_i = TCd[str(i)]
        T = TCd_i['T']
        T = T[:, ~np.isnan(T).any(axis=0)]
        if np.shape(T)[1] == 0:
            print('No Temp')
        else:
            u = np.append(u, T, axis=1)

    u = np.delete(u, 0, 1)

    for j in range(0, TCd.shape[1]):
        TCd_j = TCd[str(j)]
        Q = TCd_j['Q']
        Q = Q[:, ~np.isnan(Q).any(axis=0)]
        if np.shape(Q)[1] == 0:
            print('No Heat Flow')
        else:
            u = np.append(u, Q, axis=1)

    u = pd.DataFrame(u)

    return u, rad_surf_tot

def u_assembly_c(TCd_c, rad_surf_tot):
    rad_surf_tot = rad_surf_tot.loc[:, rad_surf_tot.any()]
    u_c = np.empty((len(rad_surf_tot), 1))  # create u matrix
    for i in range(0, TCd_c.shape[1]):
        TCd_i = TCd_c[str(i)]
        T = TCd_i['T']
        T = T[:, ~np.isnan(T).any(axis=0)]
        if np.shape(T)[1] == 0:
            print('No Temp')
        else:
            u_c = np.append(u_c, T, axis=1)

    u_c = np.delete(u_c, 0, 1)

    for j in range(0, TCd_c.shape[1]):
        TCd_j = TCd_c[str(j)]
        Q = TCd_j['Q']
        Q = Q[:, ~np.isnan(Q).any(axis=0)]
        u_c = np.append(u_c, Q, axis=1)

    u_c = pd.DataFrame(u_c)

    return u_c, rad_surf_tot


def assembly(TCd):
    """
    Description: The assembly function is used to define how the nodes in the disassembled thermal circuits
    are merged together.

    Inputs: TCd

    Outputs: AssX
    """
    TCd_last_node = np.zeros(TCd.shape[1] - 1)  # define size of matrix for last node in each TC
    TCd_element_numbers = np.arange(1, TCd.shape[1], 1)  # create vector which contains the number for each element

    # compute number of last node of each thermal circuit and input into thermal circuit sizes matrix
    for i in range(0, len([TCd_last_node][0])):
        TCd_last_node[i] = len(TCd[str(i + 1)]['A'][0]) - 1

    print(TCd_last_node)

    IA_nodes = np.arange(len(TCd[str(0)]['A'][0]))  # create vector with the nodes for inside air
    print(IA_nodes)

    # create assembly matrix
    AssX = np.zeros((len(IA_nodes), 4))  # define size of AssX matrix
    for i in range(0, len([AssX][0])):
        AssX[i, 0] = TCd_element_numbers[i]  # set first column of row to element
        AssX[i, 1] = TCd_last_node[i]  # set second column to last node of that element
        AssX[i, 2] = 0  # set third column to inside air element
        AssX[i, 3] = IA_nodes[i]  # set 4th column to element of inside air which connects to corresponding element

    AssX = AssX.astype(int)

    print(AssX)

    return AssX


def solver(TCAf, TCAc, TCAh, dt, u, u_c, t, Tisp, DeltaT, DeltaBlind, Kpc, Kph, rad_surf_tot):
    [Af, Bf, Cf, Df] = dm4bem.tc2ss(TCAf['A'], TCAf['G'], TCAf['b'], TCAf['C'], TCAf['f'], TCAf['y'])
    [Ac, Bc, Cc, Dc] = dm4bem.tc2ss(TCAc['A'], TCAc['G'], TCAc['b'], TCAc['C'], TCAc['f'], TCAc['y'])
    [Ah, Bh, Ch, Dh] = dm4bem.tc2ss(TCAh['A'], TCAh['G'], TCAh['b'], TCAh['C'], TCAh['f'], TCAh['y'])

    # Maximum time-step
    dtmax = min(-2. / np.linalg.eig(Af)[0])
    print(f'Maximum time step f: {dtmax:.2f} s')

    dtmax = min(-2. / np.linalg.eig(Ac)[0])
    print(f'Maximum time step c: {dtmax:.2f} s')

    dtmax = min(-2. / np.linalg.eig(Ah)[0])
    print(f'Maximum time step h: {dtmax:.2f} s')

    # Step response
    # -------------
    duration = 3600 * 24 * 1  # [s]
    # number of steps
    n = int(np.floor(duration / dt))

    t_ss = np.arange(0, n * dt, dt)  # time

    # Vectors of state and input (in time)
    n_tC = Af.shape[0]  # no of state variables (temps with capacity)
    # u = [To To To Tsp Phio Phii Qaux Phia]
    u_ss = np.zeros([(u.shape[1]), n])
    u_ss[0:3, :] = np.ones([3, n])
    u_ss[4:6, :] = 1

    # initial values for temperatures obtained by explicit and implicit Euler
    temp_exp = np.zeros([n_tC, t_ss.shape[0]])
    temp_imp = np.zeros([n_tC, t_ss.shape[0]])

    I = np.eye(n_tC)
    for k in range(n - 1):
        temp_exp[:, k + 1] = (I + dt * Ac) @ \
                             temp_exp[:, k] + dt * Bc @ u_ss[:, k]
        temp_imp[:, k + 1] = np.linalg.inv(I - dt * Ac) @ \
                             (temp_imp[:, k] + dt * Bc @ u_ss[:, k])

    y_exp = Cc @ temp_exp + Dc @ u_ss
    y_imp = Cc @ temp_imp + Dc @ u_ss

    fig, axs = plt.subplots(2, 1)
    # axs[0].plot(t_ss / 3600, y_exp.T, t_ss / 3600, y_imp.T)
    # axs[0].set(ylabel='$T_i$ [°C]', title='Step input: To = 1°C')

    # initial values for temperatures
    temp_exp = np.zeros([n_tC, t.shape[0]])
    temp_imp = np.zeros([n_tC, t.shape[0]])
    Tisp = Tisp * np.ones(u.shape[0])
    y = np.zeros(u.shape[0])
    y[0] = Tisp[0]
    qHVAC = 0 * np.ones(u.shape[0])

    # integration in time

    I = np.eye(n_tC)
    for k in range(u.shape[0] - 1):
        if y[k] > Tisp[k] + DeltaBlind:
            us = u_c
        else:
            us = u
        if y[k] > DeltaT + Tisp[k]:
            temp_exp[:, k + 1] = (I + dt * Ac) @ temp_exp[:, k] \
                                 + dt * Bc @ us.iloc[k, :]
            y[k + 1] = Cc @ temp_exp[:, k + 1] + Dc @ us.iloc[k + 1]
            qHVAC[k + 1] = Kpc * (Tisp[k + 1] - y[k + 1])
        elif y[k] < Tisp[k]:
            temp_exp[:, k + 1] = (I + dt * Ah) @ temp_exp[:, k] \
                                 + dt * Bh @ us.iloc[k, :]
            y[k + 1] = Ch @ temp_exp[:, k + 1] + Dh @ us.iloc[k + 1]
            qHVAC[k + 1] = Kph * (Tisp[k + 1] - y[k + 1])
        else:
            temp_exp[:, k + 1] = (I + dt * Af) @ temp_exp[:, k] \
                                 + dt * Bf @ us.iloc[k, :]
            y[k + 1] = Cf @ temp_exp[:, k + 1] + Df @ us.iloc[k]
            qHVAC[k + 1] = 0

    # plot indoor and outdoor temperature
    axs[0].plot(t / 3600, y, label='$T_{indoor}$')
    axs[0].plot(t / 3600, rad_surf_tot['To'], label='$T_{outdoor}$')
    axs[0].set(xlabel='Time [h]',
               ylabel='Temperatures [°C]',
               title='Simulation for weather')
    axs[0].legend(loc='upper right')

    # plot total solar radiation and HVAC heat flow
    Φt = rad_surf_tot.sum(axis=1)
    axs[1].plot(t / 3600, qHVAC, label='$q_{HVAC}$', linestyle='-')
    axs[1].plot(t / 3600, Φt, label='$Φ_{total}$')
    axs[1].set(xlabel='Time [h]',
               ylabel='Heat flows [W]')
    axs[1].legend(loc='upper right')
    plt.ylim(-1500, 20000)
    fig.tight_layout()

    plt.show()

    return qHVAC

def DSH(qHVAC, rad_surf_tot, Tisp):
    qHVAC_diff = np.diff(qHVAC)
    qHVAC_red = qHVAC
    for i in range(0, qHVAC_diff.shape[0]):
        a = int(math.ceil(qHVAC_diff[i]))
        if a in range(1, 10):
            break
        else:
            qHVAC_red = np.delete(qHVAC_red, 0)

    rad_surf_tot_bc = rad_surf_tot.drop(rad_surf_tot.index[range(i)])

    qHVAC_bc = np.zeros(qHVAC_red.shape[0])
    for i in range(0, qHVAC_red.shape[0]):
        if Tisp == rad_surf_tot_bc['To'][i]:
            qHVAC_bc[i] = 0
        else:
            qHVAC_bc[i] = qHVAC_red[i] / (Tisp - rad_surf_tot_bc['To'][i])

    qHVAC_bc_max = max(qHVAC_bc)
    T_diff_max = Tisp - -3
    Qmax = (qHVAC_bc_max * T_diff_max) / 1000

    return qHVAC_bc_max, Qmax
