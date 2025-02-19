# 28, Feb, 2024
# The model is for the article "Leaf-level gas exchange experiments indicate a compensation point for carbonyl sulfide"
# The model calculates fluxes and mole fractions of H2O, CO2, and COS for several layers inside a leaf.
# This code is based on Python 3. 

# Authors: A.Cho, L.M.J.Kooijmans, S.M.Diever, M.Wassenaar, G.Koren, M.E.Popa, S.L.Baartman, L.Mossink, S.van Heuven, and M.C.Krol
# Correspondence: Ara Cho (ara.cho@wur.nl)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def forward_model(Tleaf_M, flow_M, h2o_in_M, co2_in_Mp, cos_in_Mp, gbw_M, pressure_M):
    
    ##### Optimized state variables ####
    gsw_M    = 0.61   # Average of stomatal conductance from experimental dataset [mol/m2/s]
    teq_M    = 35.0   # Optimum temperature for Carbonic Anhydrase (CA) [C]
    act_e_M  = 50.5   # Activity energy for RuBisCO [kJ/mol]
    mcos_M   = 21.9   # Slope of COS compensation point [pmol/mol/K]
    vmaxca_M = 0.159  # Maximum velocity of CA [mol/m2/s]
                      # Sunflower 2: 0.159, Sunflower 3: 0.140, Sunflower 4: 0.0161
    vmax_M   = 93.482 # Maximum velocity of RuBisCO [mol/m2/s]
                      # Sunflower 2 : 93.482, Sunflower 3: 85.128, Sunflower 4: 81.103
        
    
    ##### Constant ####
    # Global
    gas_R = 8.314           # Ideal gas constant [J/mol/K]
    air_mv = 0.0248         # Dry air molar volume [m3/mol] @ 25C and 1 bar
    S = 0.0009              # Leaf Area [m2]
    
    tc = Tleaf_M + 273.15
    
    
    ################# Sub-model 1. Water Vapor ###################
    # Constant 
    gmw = 10.               # mesophyll conductance of water vapor [mol/m2/s]
    
    # Saturated water vapor at a leaf temperature [Pa]
    es_Tl = 613.5*np.exp(Tleaf_M/(Tleaf_M+240.97)*17.502) 
    
    # Water Vapor mole fraction in mesophyll cells [mmol/mol]. Assuemd as saturated vapor.
    wc = 1000.*air_mv*es_Tl/(8.314*tc)  
    
    # Water Vapor mole fraction in atmosphere [mmol/mol]
    wa = (flow_M*h2o_in_M+S*gbw_M*((S*gsw_M*gmw*wc)/(gsw_M+gmw))/(S*gbw_M+S*gsw_M-S*(gsw_M**2)/(gsw_M+gmw)))\
     /(S*gbw_M+flow_M-(S**2)*(gbw_M**2)/(S*gbw_M+S*gsw_M-S*(gsw_M**2)/(gsw_M+gmw)))

    # Water Vapor mole fraction in boundary layer [mmol/mol]
    wb = (S*gbw_M*wa+S*gsw_M*gmw*wc/(gsw_M+gmw))/(S*gbw_M+S*gsw_M-S*(gsw_M**2)/(gsw_M+gmw))
    
    # Water Vapor mole fraction in internal cells [mmol/mol]
    ws = (S*gsw_M*wb+S*gmw*wc)/(S*gsw_M+S*gmw)
    
    # Relative humidity inside a leaf [%]
    RH_s = 100*(((gas_R*tc*ws)/(1000*air_mv))/es_Tl)
    
    # Transpiration [mol/m2/s]
    wvflux = - flow_M*(h2o_in_M-wa)*1e-3/(S)
    
    
    ################# Sub-model 2. CO2 ###################
    # Constant
    spfy_val = 3416.              # Specificity factor between CO2 and O2 at 298 K [-]
    zko_val = 33000.              # Michaelis-Menten constant for oxygenation at 298 K [Pa]
    Rd_val = 2.0                  # Dark repiration at 298 K [umol/m2/s]
    zkc_val = 46
    
    po2m = 20900.                 # Parial pressure of O2 [Pa]  
    
    # gs and gb of CO2 [mol/m2/s]
    gs_co2 = gsw_M/1.6
    gb_co2 = gbw_M/1.4
     
    qt = 0.1*(tc-298)             # Q10 at Tleaf with reference temperature 298 K [-]
    zkc= zkc_val*(2.1**qt)        # Michaelis constant of CO2 at Tleaf [-]
    zko = zko_val * (1.2**qt)     # Michaelis constant of O2 at Tleaf [-]
    
    act_e_co2 =act_e_M*1000.      # Activation energy for RuBisCo (unit conversion) [J/mol]
    
    # Vmax of RuBisCo with its temperature function at Tleaf  
    vmaxts = vmax_M*np.exp((tc-298.)*act_e_co2/(298.*gas_R*(Tleaf_M+273.15)))
  
    spfy = spfy_val * (0.57**qt)  # Partitioning of RuBP to the Caboxylase or Oxygenase [-]
                                  
    gamma = 0.5*(po2m/spfy)       # CO2 compensation point [Pa]  
    Rd = Rd_val * 2.0**qt        # CO2 respiration [umol/m2/s]
    
    # Analytical solving
    zk = zkc*(1+po2m/zko)
    p =  1e-6*pressure_M

    co2_in_M = co2_in_Mp
    
    T1 = (co2_in_M*flow_M/S)/(gb_co2+flow_M/S)
    T2 = 1+gs_co2/gb_co2-wvflux/(2*gb_co2)-gb_co2/(gb_co2+flow_M/S)

    a = (gs_co2**2)*p/(gb_co2*T2) + gs_co2*p*wvflux/(2*gb_co2*T2) - gs_co2*p - p*wvflux*gs_co2/(2*gb_co2) - p*(wvflux**2)/(4*gb_co2*T2) - wvflux*p/2
    b = gs_co2*p*T1/T2 + (gs_co2**2)*zk/(gb_co2*T2) + gs_co2*zk*wvflux/(2*gb_co2*T2) - gs_co2*zk - wvflux*p*T1/(2*T2) - (wvflux*gs_co2*zk)/(2*gb_co2*T2) - (wvflux**2)*zk/(4*gb_co2*T2) - vmaxts*p - wvflux*zk/2 + Rd*p
    c = gs_co2*zk*T1/T2 - wvflux*zk*T1/(2*T2) + vmaxts*gamma + Rd*zk
    
    # CO2 mole fraction in internal cells [umol/mol]. 
    cs_co2 = (-b-np.sqrt(b**2-4*a*c))/(2*a)
    
    # CO2 mole fraction in boundary layer [umol/mol]. 
    cb_co2 = (T1+cs_co2*gs_co2/gb_co2 + cs_co2*wvflux/(2*gb_co2))/T2
    
    # CO2 mole fraction in atmosphere [umol/mol]. 
    ca_co2 = (gb_co2*cb_co2+flow_M*co2_in_M/S)/(gb_co2+flow_M/S)    
    ca_co2_dry = ca_co2/(1-(h2o_out_M/1000.)) 
    
    # CO2 flux [mol/m2/s]
    flux_co2 = flow_M*(co2_in_Mp-ca_co2_dry)/S
    
    # Partial pressure of internal CO2 [Pa]
    pcs_co2 = cs_co2*p 
    
    # internal conductance of CO2 [mol/m2/s]
    gi_co2 = vmaxts*(pcs_co2-gamma)/(pcs_co2+zk)/cs_co2
    
    
    
    ################# Sub-model 3. COS ###################
    # Constant
    
    ha = 40000.             # Activation energy for CA [J/mol]
    heq = 100000.           # Enthalphy change for CA [J/mol]
    teq_abs = teq_M+273.15    # Optimum temperature [K]
    
    # gs and gb of COS [mol/m2/s]
    gs_cos = gsw_M/1.94
    gb_cos = gbw_M/1.56
    
    # Internal conductance for COS [mol/m2/s]
    cosgm_t = tc*np.exp(-ha/(gas_R*tc))/(1.+np.exp(-(heq/gas_R)\
            *(1./tc-1./teq_abs)))
    cosgm_max = teq_abs*np.exp(-ha/(gas_R*teq_abs))/(1.+np.exp(-(heq/gas_R)\
            *(1./teq_abs-1./teq_abs)))
    gi_cos = (cosgm_t/cosgm_max)*vmaxca_M 
    
    # Compensation point for COS [pmol/mol]
    cc_cos_pre = mcos_M*(tc-289.36)
    cc_cos =(np.where((cc_cos_pre >= 0.),cc_cos_pre,0.))

    cos_in_M = cos_in_Mp * (1-(h2o_in_M/1000.))
    
    T = gs_cos+gi_cos+wvflux/2.
    T2 = gb_cos+flow_M/S
    
    # COS mole fraction in boundary layer [pmol/mol]. 
    cb_cos = ((gb_cos*flow_M*cos_in_M/S)/T2 + gs_cos*gi_cos*cc_cos/T + wvflux*gi_cos*cc_cos/(2*T))/(-(gb_cos**2)/T2 + gb_cos + gs_cos - (gs_cos**2)/T + gs_cos*wvflux/(2*T)-wvflux/2-wvflux*gs_cos/(2*T)+(wvflux**2)/(4*T) )
    
    # COS mole fraction in internal cells [pmol/mol]. 
    ca_cos = (gb_cos*cb_cos+flow_M*cos_in_M/S)/T2
    
    # COS mole fraction in atmosphere [pmol/mol]. 
    cs_cos = (gs_cos*cb_cos - (cb_cos*wvflux/2) + gi_cos*cc_cos)/(gs_cos+gi_cos+wvflux/2)

    # COS flux [mol/m2/s]
    ca_cos_dry = ca_cos/(1-(h2o_out_M/1000.)) 
    
    flux_cos = flow_M*(cos_in_Mp-ca_cos_dry)/S

    ##### Output data ####
    # wvflux                         = water vapor flux [mol/m2/s]
    # wa, wb, ws, wc                 = water vapor mole fraction in atmosphere, boundary layer, intercelluar space, and mesophyll cells [mmol/mol]
    # flux_co2                       = CO2 flux [mol/m2/s]
    # ca_co2, cb_co2, cs_co2         = CO2 mole fraction in atmosphere, boundary layer, and intercellular space
    # flux_cos                       = COS flux [mol/m2/s]
    # ca_cos, cb_cos, cs_cos, cs_cos = COS mole fraction in atmosphere, boundary layer, intercellular space, and mesophyll cells [pmol/mol]
    # RH_s                           = intercellular relative humidity [%]
    # gi_co2                         = mesophyll conductance of CO2 [mol/m2/s]
    # gi_cos                         = mesophyll conductance of cos [mol/m2/s]
    
    return(wvflux, wa, wb, ws, wc, flux_co2, ca_co2, cb_co2, cs_co2, flux_cos, ca_cos, cb_cos, cs_cos, cc_cos, gi_cos, gi_co2, RH_s)

# Test the model

# Input variables 
Tleaf    = 25.7      # Leaf temperature [C]
flow_obs = 0.00035   # Air flow rate [mol/s]
h2o_in   = 15        # Ingoing mole fraction of water vapor [mmol/mol]
co2inflow = 400      # Ingoing mole fraction of CO2 [umol/mol]
cosinflow = 1000     # Ingoing mole fraction of COS [pmol/mol]
gbw       = 2.44     # Boundary conductance of water vapor [mol/m2/s]
pressure = 103100    # air pressure in a leaf cuvette [Pa]

est_wvflux,  est_wa, est_wb, est_ws, est_wc,\
est_co2flux, est_co2_ca, est_co2_cb, est_co2_cs,\
est_cosflux, est_cos_ca, est_cos_cb, est_cos_cs, est_cos_cc,\
est_RH_s, est_gi_co2, est_gi_cos\
= forward_model(Tleaf, flow_obs, h2o_in, co2inflow, cosinflow, gbw, pressure)

print('Estimated H2O flux = %.4f mol/m2/s'%est_wvflux)
print('Estimated CO2 flux = %.2f mol/m2/s'%est_co2flux)
print('Estimated COS flux = %.2f mol/m2/s'%est_cosflux)

print('Estimated mole fractions in atmosphere (A), boundary (B), intercellular (C), and mesophyll (D)')
print('H2O A = %.2f mmol/mol, B= %.2f mmol/mol, C= %.2f mmol/mol, D= %.2f mmol/mol'%(est_wa, est_wb, est_ws, est_wc))
print('CO2 A = %.2f umol/mol, B= %.2f umol/mol, C= %.2f umol/mol'%(est_co2_ca, est_co2_cb, est_co2_cs))
print('COS A = %.2f pmol/mol, B= %.2f pmol/mol, C= %.2f pmol/mol, D= %.2f pmol/mol'%(est_cos_ca, est_cos_cb, est_cos_cs, est_cos_cc))
