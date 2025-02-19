# Optimization of the conductance model combined H2O, CO2, and COS exchanges in a leaf
# Containing 100 simulations by the Monte-Carlo method.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

######## Options ########
# Select the model for the COS compensation point (fuction to leaf temperature)
#mode = 'S1' # No COS compensation point
mode = 'S2' # a linear ### THE MODE WHICH WE SELECT ###
#mode = 'S3' # a Arrhenius function with the reference temperature 20 C
#mode = 'S4' # a Arrhenius function with the reference temperature 25 C

# Select the operatiob mode
iter_mode = 'single' # Analytical solution
#iter_mode = 'multiple' # Monte-Carlo approach

if(iter_mode == 'single'):
    tot_num = 1
elif(iter_mode == 'multiple'):
    tot_num = 100 # Number of ensemble members = 100
    
# Weights
w_wv = 3.236 ; w_co2 =  7.1647; w_cos = 0.03366 ; w_bg=1.0 # Feb 2025

nplt = 3 # the number of plants
########################

# Read a file of experiment dataset
file = './Photosynthesis_experiment_leaf_2022.csv'
leaf = pd.read_csv(file)

Tleaf       = np.array(leaf.Tleaf)       # Leaf temperature [C]
flow_obs    = np.array(leaf.airflow)     # Air flow rate [mol/s]
h2o_in      = np.array(leaf.h2o_in)      # Ingoing mole fraction of H2O [mmol/mol]
h2o_out     = np.array(leaf.h2o_out)     # Outgoing mole fraction of H2O [mmol/mol]
co2inflow   = np.array(leaf.co2_in)      # Ingoing mole fraction of CO2 [umol/mol]
co2outflow  = np.array(leaf.co2_out)     # Outgoing mole fraction of CO2 [umol/mol]
cosinflow   = np.array(leaf.cos_in)      # Ingoing mole fraction of COS [pmol/mol]
cosoutflow  = np.array(leaf.cos_out)     # Outgoing mole fraction of COS [pmol/mol]

gsw         = np.array(leaf.gsw)         # Stomatal conductance of water vapor [mol/m2/s]
gbw         = np.array(leaf.gbw)         # Boundary conductance of water vapor [mol/m2/s]
pressure    = np.array(leaf.pressure)    # air pressure in a leaf cuvette [Pa]
plant       = np.array(leaf.plant)       # Plant name [string]
gsw_unc     = np.array(leaf.gsw_unc)     # Uncertainty of stomatal conductance 
h2o_out_unc = np.array(leaf.h2o_out_unc) # Uncertainty of outgoing mole fraction of H2O (Standard deviation)
co2_out_unc = np.array(leaf.co2_out_unc) # Uncertainty of outgoing mole fraction of CO2 (Standard deviation)
cos_out_unc = np.array(leaf.cos_out_unc) # Uncertainty of outgoing mole fraction of COS (Standard deviation)

ndata = len(Tleaf)

######## forward model ########
def forward_model(state, mode_mcos, Tleaf_M, flow_M, h2o_in_M, co2_in_Mp, cos_in_Mp, gbw_M, pressure_M, plant_M):
    
    ndata = len(Tleaf_M)
    
    ##### Optimized state variables ####
    
    # Arrange state
    
    gsw_M = state[0:ndata]
    
    if (mode_mcos !='S1'):  # Models S2, S3, S4
        nothers = 3
        teq_M, act_e_M, mcos_M = state[ndata:ndata+nothers]
    else:                   # Model S1
        nothers = 2
        teq_M, act_e_M = state[ndata:ndata+nothers]
    
    vmaxca_M = np.where((plant_M =='sunflower_1'),state[ndata+nothers],\
                      (np.where((plant_M =='sunflower_2_leaf2'),state[ndata+nothers+1],\
                      (np.where((plant_M =='sunflower_3'),state[ndata+nothers+2],np.nan)))))
    vmax_M = np.where((plant_M =='sunflower_1'),state[ndata+nothers+3],\
                      (np.where((plant_M =='sunflower_2_leaf2'),state[ndata+nothers+4],\
                      (np.where((plant_M =='sunflower_3'),state[ndata+nothers+5],np.nan)))))
        
    
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
    Rd = Rd_val * 2.13**qt        # CO2 respiration [umol/m2/s]

    co2_in_M = co2_in_Mp
    
    # Analytical solving
    zk = zkc*(1+po2m/zko)
    p =  1e-6*pressure_M
    
    f1 = flow_M*co2_in_M*gs_co2/(S*gb_co2+flow_M)
    f2 = (gs_co2/gb_co2+1-(S*gb_co2)/(S*gb_co2+flow_M)) 
    
    a = -(p*gs_co2**2/(gb_co2*f2))+gs_co2*p
    b = gs_co2*zk+vmaxts*p-Rd*p-f1*p/f2-(gs_co2**2)*zk/(gb_co2*f2)
    c = -Rd*zk-f1*zk/f2-vmaxts*gamma
    
    # CO2 mole fraction in internal cells [umol/mol]. 
    cs_co2 = ((-b+np.sqrt(b**2-(4*a*c)))/(2*a))
    
    # CO2 mole fraction in boundary layer [umol/mol]. 
    cb_co2 = cs_co2+ (vmaxts*(p*cs_co2-gamma)/(gs_co2*(cs_co2*p+zk)))-Rd/gs_co2
    
    # CO2 mole fraction in atmosphere [umol/mol]. 
    ca_co2 = cb_co2 + gs_co2*cb_co2/gb_co2 - gs_co2*cs_co2/gb_co2
    
    # CO2 flux [mol/m2/s]
    flux_co2 = flow_M*(co2_in_M-ca_co2)/S
    
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
    
    if (mode_mcos == 'S1'):
        cc_cos_pre = 0.
    elif(mode_mcos == 'S2'):
        cc_cos_pre = mcos_M*(tc-289.36)
    elif(mode_mcos == 'S3'):
        cc_cos_pre = 55.0*np.exp((mcos_M*1000.)*(tc-293.0)/(293.0*gas_R*tc))
    elif(mode_mcos == 'S4'):
        cc_cos_pre = 138.7*np.exp((mcos_M*1000.)*(tc-298.0)/(298.0*gas_R*tc))
        
    cc_cos =(np.where((cc_cos_pre >= 0.),cc_cos_pre,0.))
        
    cos_in_M = cos_in_Mp * (1-(h2o_in_M/1000.))
    
    # COS mole fraction in boundary layer [pmol/mol]. 
    cb_cos = (cos_in_M*flow_M/(gb_cos*S+flow_M)+((gs_cos*gi_cos*cc_cos)/(gb_cos*(gs_cos+gi_cos))))/(-gb_cos*S/(gb_cos*S+flow_M)+1+gs_cos/gb_cos-(gs_cos**2/(gb_cos*(gs_cos+gi_cos)))) #ppt
    
    # COS mole fraction in internal cells [pmol/mol]. 
    cs_cos= (gs_cos*cb_cos+gi_cos*cc_cos)/(gs_cos+gi_cos)
    
    # COS mole fraction in atmosphere [pmol/mol]. 
    ca_cos= cb_cos+gs_cos*cb_cos/gb_cos-gs_cos*cs_cos/gb_cos

    # COS flux [mol/m2/s]
    flux_cos = flow_M*(cos_in_M-ca_cos)/S

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
########################


# Monte Carlo method 
# 100 distinct ensemble members, introducing random noise.

def noise_generator(value, error):
    random = np.random.normal(size=len(value))
    noisy_out = value + error * random
    return(noisy_out)

# Generate noise for observed mole fraction of COS, CO2, and H2O
def random_obs(h2o_obs,h2o_err, co2_obs,co2_err,cos_obs, cos_err): 
    cos_random = noise_generator(cos_obs,cos_err)
    co2_random = noise_generator(co2_obs,co2_err) 
    h2o_random = noise_generator(h2o_obs,h2o_err)

    return(cos_random, co2_random, h2o_random)

# Generate noise for state variables
def random_init(state, state_error):
    init_random = noise_generator(state,state_error)
    
    return(init_random)



# State variables for single value across three plants
# Teq, d.Ha for RuBisCO, mcos for cos compensation point

if(mode == 'S1'):    
    state_init_others = np.array([ 30, 60.])
    state_error_others = np.array([ 15., 12.])
elif(mode == 'S2'):   
    state_init_others = np.array([ 30, 60., 16.2])
    state_error_others = np.array([ 15., 12., 16.2])
else:   
    state_init_others = np.array([ 30, 60.,134.3])
    state_error_others = np.array([ 15., 12., 60.0])

nothers = len(state_init_others)

# State variables for multiple value
# Vmax of CA
state_init_vmaxca = np.full((nplt), 0.125)
state_error_vmaxca = np.full((nplt), 0.06)

# Vmax of RuBisCO
state_init_vmax = np.full((nplt), 90)
state_error_vmax = np.full((nplt), 20)

# Combine all state variables
state_init = np.concatenate((gsw, state_init_others, state_init_vmaxca,\
                             state_init_vmax), axis=0)
state_error = np.concatenate((gsw_unc, state_error_others, state_error_vmaxca,\
                             state_error_vmax),axis=0)


# Minimize the cost function
def min_func(state, state_err, mode_mcos, wt_wv, wt_co2, wt_cos, wt_bg, h2o_obs, h2o_err, co2_obs, co2_err, cos_obs, cos_err, \
             Tleaf_M, flow_M, h2o_in_M, co2_in_M, cos_in_M, plant_M, gbw_M, pressure_M):
    
    # Description of input:
    # State, state_err : State variables and their errors
    # wt_wv, wt_co2, wt_cos, wt_bg : weight of H2O, CO2, COS, and the state term
    # h2o_obs, h2o_err : Observed mole fraction of H2O and its error
    # co2_obs, co2_err : Observed mole fraction of CO2 and its error
    # cos_obs, cos_err : Observed mole fraction of COS and its error
    # The rests are variabes for the forward model
    
    # Forward model run
    est_wvflux,  est_wa, est_wb, est_ws, est_wc,\
    est_co2flux, est_co2_ca, est_co2_cb, est_co2_cs,\
    est_cosflux, est_cos_ca, est_cos_cb, est_cos_cs, est_cos_cc,\
    est_RH_s, est_gi_co2, est_gi_cos\
    = forward_model(state, mode_mcos, Tleaf, flow_obs, h2o_in, co2inflow, cosinflow, gbw, pressure, plant)
    
    
    # Calculate the cost for Water vapor
    obs=h2o_obs
    error_obs =h2o_err
    obs_dev = (est_wa - obs)**2
    obs_wv_cost = np.nansum(obs_dev/(wt_wv*error_obs**2))
    num_obs = np.count_nonzero(~np.isnan(obs_dev))
    
    # Calculate the cost for CO2
    obs=co2_obs
    error_obs =co2_err
    obs_dev = (est_co2_ca - obs)**2
    obs_co2_cost = np.nansum(obs_dev/(wt_co2*error_obs**2))
    num_obs = np.count_nonzero(~np.isnan(obs_dev))
    
    # Calculate the cost for COS
    obs=cos_obs
    error_obs =cos_err
    obs_dev = (est_cos_ca - obs)**2
    obs_cos_cost = np.nansum(obs_dev/(wt_cos*error_obs**2))
    num_obs = np.count_nonzero(~np.isnan(obs_dev))
    
    
    # Cost for state variables 
    state_dev = (state-state_init)**2
    state_cost = np.nansum(state_dev/(wt_bg*state_err**2))

    tot_cost = state_cost + (obs_wv_cost+ obs_co2_cost + obs_cos_cost)
    
    
    return(tot_cost)


def objective(x,a,b):
    return a*x+b

def stat_index(obs, mod):
    nas = np.logical_or(np.isnan(obs)==True, np.isnan(mod)==True)
    x, y  =obs[~nas], mod[~nas]
    popt, _ = curve_fit(objective, x, y)
    a,b = popt
    residuals = y- objective(x, *popt)
    ss_res = np.nansum(residuals**2)
    ss_tot = np.nansum((y-np.nanmean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(mean_squared_error(x, y))
    diff = y-x
    mbe = np.nanmean(diff)
    
    return(rmse, mbe)

# Calculate final cost
def cal_func(state, state_err, mode_mcos, wt_wv, wt_co2, wt_cos, wt_bg, h2o_obs, h2o_err, co2_obs, co2_err, cos_obs, cos_err, \
             Tleaf_M, flow_M, h2o_in_M, co2_in_M, cos_in_M, plant_M, gbw_M, pressure_M):
    
    # Description of input:
    # State, state_err : State variables and their errors
    # wt_wv, wt_co2, wt_cos, wt_bg : weight of H2O, CO2, COS, and the state term
    # h2o_obs, h2o_err : Observed mole fraction of H2O and its error
    # co2_obs, co2_err : Observed mole fraction of CO2 and its error
    # cos_obs, cos_err : Observed mole fraction of COS and its error
    # The rests are variabes for the forward model
    
    # Forward model run
    est_wvflux,  est_wa, est_wb, est_ws, est_wc,\
    est_co2flux, est_co2_ca, est_co2_cb, est_co2_cs,\
    est_cosflux, est_cos_ca, est_cos_cb, est_cos_cs, est_cos_cc,\
    est_RH_s, est_gi_co2, est_gi_cos\
    = forward_model(state, mode_mcos, Tleaf, flow_obs, h2o_in, co2inflow, cosinflow, gbw, pressure, plant)
    
    
    # Calculate the cost for Water vapor
    obs=h2o_obs
    error_obs =h2o_err
    obs_dev = (est_wa - obs)**2
    obs_wv_cost = np.nansum(obs_dev/(wt_wv*error_obs**2))
    num_obs = np.count_nonzero(~np.isnan(obs_dev))
    chi_wv = obs_wv_cost/num_obs
    rmse_wv, mbe_wv = stat_index(obs, est_wa)
    
    # Calculate the cost for CO2
    obs=co2_obs
    error_obs =co2_err
    obs_dev = (est_co2_ca - obs)**2
    obs_co2_cost = np.nansum(obs_dev/(wt_co2*error_obs**2))
    num_obs = np.count_nonzero(~np.isnan(obs_dev))
    chi_co2 = obs_co2_cost/num_obs
    rmse_co2, mbe_co2 = stat_index(obs, est_co2_ca)
    
    # Calculate the cost for COS
    obs=cos_obs
    error_obs =cos_err
    obs_dev = (est_cos_ca - obs)**2
    obs_cos_cost = np.nansum(obs_dev/(wt_cos*error_obs**2))
    num_obs = np.count_nonzero(~np.isnan(obs_dev))
    chi_cos = obs_cos_cost/num_obs
    rmse_cos, mbe_cos = stat_index(obs, est_cos_ca)
    
    # Cost for state variables 
    state_dev = (state-state_init)**2
    state_cost = np.nansum(state_dev/(wt_bg*state_err**2))
    num_state = len(state)
    chi_state = state_cost/num_state

    tot_cost = state_cost + (obs_wv_cost+ obs_co2_cost + obs_cos_cost)
    
    print('Each cost: state = %.2f, H2O = %.2f, CO2 = %.2f, COS = %.2f, J_tot = %.2f'%\
          (state_cost, obs_wv_cost, obs_co2_cost, obs_cos_cost, tot_cost))
    print('chi^2    : state = %.2f, H2O = %.2f, CO2 = %.2f, COS = %.2f'%(chi_state, chi_wv, chi_co2, chi_cos))
    print('RMSE     : H2O = %.2f, CO2 = %.2f, COS = %.2f'%(rmse_wv, rmse_co2, rmse_cos))
    print('MBE      : H2O = %.2f, CO2 = %.2f, COS = %.2f'%(mbe_wv, mbe_co2, mbe_cos))
    return(tot_cost)



ndata = len(Tleaf)

opt_gs = np.empty([ndata,tot_num])*np.nan
opt_teq = np.empty([tot_num])*np.nan
opt_e_co2 = np.empty([tot_num])*np.nan
opt_mcos = np.empty([tot_num])*np.nan
opt_vmaxca = np.empty([3,tot_num])*np.nan
opt_vmax = np.empty([3,tot_num])*np.nan

# Bounds of the states
state_opt=[]

# gsw
bnds_gsw = [(0.0,3.0)]*ndata
# Variables having a single value

if(mode == 'S1'):
    bnds_others = [(1.0,60.),(1.0,300.0)]
elif(mode == 'S2'):
    bnds_others = [(1.0,60.),(1.0,300.0),(0.0,100.0)] 
elif(mode == 'S3' or mode == 'S4'):
    bnds_others = [(1.0,60.),(1.0,300.0),(0.0,400.0)] 
    
# Variables having multiple values
bnds_vmaxca = [(0.01,0.5)]*3
bnds_vmax = [(1.0,200)]*3
# Total
bnds = bnds_gsw+bnds_others+bnds_vmaxca+bnds_vmax



# Optimize parameters for 100 times
for inum in range(0, tot_num):
    print(inum+1, '/',tot_num, '...')
    
    
    if(iter_mode =='single'):
        
        cosoutflow_rd = cosoutflow
        co2outflow_rd = co2outflow
        h2o_out_rd = h2o_out
        state_init_random = state_init
        
    elif(iter_mode =='multiple'):
        
        # Calculate random noises for observations and state variables
        cosoutflow_rd, co2outflow_rd, h2o_out_rd = random_obs(h2o_out, h2o_out_unc, co2outflow,co2_out_unc,cosoutflow, cos_out_unc)
        state_init_random = random_init(state_init, state_error)
        
        
    # Optimize parameters using SLSQP method
    Opt = optimize.minimize(min_func, state_init_random, args = (state_error, mode, w_wv, w_co2, w_cos, w_bg, \
                                                          h2o_out_rd, h2o_out_unc, \
                                                          co2outflow_rd, co2_out_unc, \
                                                          cosoutflow_rd, cos_out_unc, \
                                                          Tleaf, flow_obs, h2o_in, co2inflow, cosinflow, plant, gbw,pressure), \
                            method='SLSQP', bounds=bnds)
    
    opt_result= Opt.x   
    
    
    # Arrange the optimized results to array
    opt_gs[:,inum] = opt_result[0:ndata]
    
    if(mode=='S1'):
        opt_teq[inum], opt_e_co2[inum] = opt_result[ndata:ndata+nothers]
        opt_others = np.array([opt_teq[inum], opt_e_co2[inum]])
    else:
        opt_teq[inum], opt_e_co2[inum], opt_mcos[inum] = opt_result[ndata:ndata+nothers]
        opt_others = np.array([opt_teq[inum], opt_e_co2[inum], opt_mcos[inum]])
        
    opt_vmaxca[0,inum], opt_vmaxca[1,inum], opt_vmaxca[2,inum] = opt_result[ndata+nothers:ndata+nothers+nplt]
    opt_vmax[0,inum], opt_vmax[1,inum], opt_vmax[2,inum] = opt_result[ndata+nothers+nplt:ndata+nothers+nplt*2]
    
    state_opt = np.concatenate((opt_gs[:,inum], opt_others,\
                                opt_vmaxca[:,inum],\
                                opt_vmax[:,inum]), axis=0)

    cost_prior = min_func(state_init_random,state_error, mode, w_wv, w_co2, w_cos, w_bg, \
                        h2o_out, h2o_out_unc, co2outflow,co2_out_unc, cosoutflow, cos_out_unc,\
                        Tleaf, flow_obs, h2o_in, co2inflow, cosinflow, plant, gbw,pressure)
    cost_opt = min_func(state_opt, state_error, mode, w_wv, w_co2, w_cos, w_bg, \
                        h2o_out, h2o_out_unc, co2outflow,co2_out_unc, cosoutflow, cos_out_unc,\
                        Tleaf, flow_obs, h2o_in, co2inflow, cosinflow, plant, gbw,pressure)
    
    
    print('Cost prior = %.2f, Posterior = %.2f'%(cost_prior, cost_opt))
    
    if(iter_mode=='single'):
        print('Results for Prior')
        cost_result = cal_func(state_init_random, state_error, mode, w_wv, w_co2, w_cos, w_bg, \
                        h2o_out, h2o_out_unc, co2outflow,co2_out_unc, cosoutflow, cos_out_unc,\
                        Tleaf, flow_obs, h2o_in, co2inflow, cosinflow, plant, gbw,pressure)
        print('   ')
        print('Results for Posterior')
        cost_result = cal_func(state_opt, state_error, mode, w_wv, w_co2, w_cos, w_bg, \
                        h2o_out, h2o_out_unc, co2outflow,co2_out_unc, cosoutflow, cos_out_unc,\
                        Tleaf, flow_obs, h2o_in, co2inflow, cosinflow, plant, gbw,pressure)
    
    # Filtering unvalid results 
    if(np.nanmin(opt_gs[:,inum])<0.):
        print('filter... due to the negative gsw: gsw = %.2f'%(np.nanmin(opt_gs[:,inum])))
        opt_teq[inum]=opt_teq[inum]*np.nan
        opt_e_co2[inum]=opt_e_co2[inum]*np.nan
        opt_gs[:,inum]=opt_gs[:,inum]*np.nan
        opt_vmaxca[:,inum]=opt_vmaxca[:,inum]*np.nan
        opt_vmax[:,inum]=opt_vmax[:,inum]*np.nan
        if(mode!='S1'):
            print('filter... due to the negative m_cos = %.2f'%(np.nanmin(opt_mcos[:,inum])))
            opt_mcos[inum]=opt_mcos[inum]*np.nan

# Save the optimized dataset
stnum = 0
ednum = stnum + tot_num
np.savetxt('opt_data_posterior_%s_%02d_%02d_gsw.csv'%(mode, stnum, ednum), opt_gs)
np.savetxt('opt_data_posterior_%s_%02d_%02d_others.csv'%(mode, stnum, ednum), (opt_teq, opt_mcos, opt_e_co2))
np.savetxt('opt_data_posterior_%s_%02d_%02d_vmaxca.csv'%(mode, stnum, ednum), (opt_vmaxca))
np.savetxt('opt_data_posterior_%s_%02d_%02d_vmax.csv'%(mode, stnum, ednum), (opt_vmax))


######### Distribution for the multiple runs ########
nv = nothers #Single variable
nv_mul = 2 # Plant dependent variable
mu = np.empty([nv+nv_mul,3])*np.nan
sigma = np.empty([nv+nv_mul,3]) *np.nan

num_bins = 30
x_data = np.empty([5,tot_num])
x_data[0,:]= opt_teq
x_data[1,:]= opt_e_co2
if(mode!='S1'):
    x_data[2,:]= opt_mcos
    label = ['$T_{eq}$', '$d.H_{a,RuB}$', '$m_{COS}$']
    if(mode=='S2'):
        v_min = [10,-50, -50]
        v_max = [60,150, 100]
    else:
        v_min = [10,-50, -10]
        v_max = [60,150, 30]
else:
    v_min = [10,-50]
    v_max = [60,150]
    label = ['$T_{eq}$', '$d.H_{a,RuB}$']
if(mode!='S1'):
    fig, axess = plt.subplots(nrows=1, ncols=3,figsize=[12,4])
else:
    fig, axess = plt.subplots(nrows=1, ncols=2,figsize=[8,4])
fig.subplots_adjust(wspace=0.5, hspace=0.5)

for ist, ax in enumerate(axess.flatten()):
    x = x_data[ist,:]
    mu[ist,:] = np.nanmean(x)   
    sigma[ist,:] = np.nanstd(x)
    
    n, bins, patches = ax.hist(x, num_bins, density=True,label='hist', range= (v_min[ist], v_max[ist]))
    mu_pri = state_init[ndata+ist]
    sigma_pri = state_error[ndata+ist] 
    
    # add a 'prior' line
    y_pri = ((1 / (np.sqrt(2 * np.pi) * sigma_pri)) *
    np.exp(-0.5 * (1 / sigma_pri * (bins - mu_pri))**2))
    ax.plot(bins, y_pri, '-',color = 'black',label='prior')
    
    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * sigma[ist,0])) *
    np.exp(-0.5 * (1 / sigma[ist,0] * (bins - mu[ist,0]))**2))
    ax.plot(bins, y, '-', color= 'red', label='post')
    
    ax.set_title(r'%s $\mu=$%.1f, $\sigma=$%.1f'%(label[ist], mu[ist,0],sigma[ist,0]), fontsize=18)
    ax.tick_params(axis='x',labelsize=15)
    ax.tick_params(axis='y',labelsize=15)
    h1,l1 = ax.get_legend_handles_labels()


num_bins = 30

for istat in range(0, 2):
    print(istat)
    
    if(istat==0): #Vmaxca
        x_data = opt_vmaxca
        v_min = 0
        v_max = 0.3
        init_no = ndata+nv
        title = 'Vmax of CA [mol $m^{-2}$ $s^{-1}$]'
    elif(istat==1): #vmax Rubisco
        x_data = opt_vmax
        v_min = 0
        v_max = 150
        init_no = ndata+nv+3
        title = 'Vmax of Rubisco [$\mu$mol $m^{-2}$ $s^{-1}$]'
    
    
    fig, axess = plt.subplots(nrows=1, ncols=3,figsize=[13,4])
    fig.subplots_adjust(top=0.8,bottom=0.23, wspace=0.25, hspace=0.0, left=0.09)

    for ist, ax in enumerate(axess.flatten()):
  
        plt.suptitle('x = '+title, fontsize=19)
        x = x_data[ist,:]
        mu[nv+istat,ist] = np.nanmean(x)
        
        sigma[nv+istat,ist] = np.nanstd(x)
        
        if(istat==2):
            print('Vmax_CA/Vmax_Rub')
            #print(vmax_f[ist]/vmaxca_f[ist])
            rat_data = opt_vmaxca[ist,:]/opt_vmax[ist,:]*1e6
            
            print('Mean = ', np.nanmean(rat_data))
            print('STD = ', np.nanstd(rat_data))
        
        if(np.isnan(np.nanmean(x))==False):
            n, bins, patches = ax.hist(x, num_bins, density=True,label='hist', range= (v_min, v_max))
     
            mu_pri = state_init[init_no]
            sigma_pri = state_error[init_no] 
            print(mu_pri, sigma_pri)
    
            # add a 'prior' line
            y_pri = ((1 / (np.sqrt(2 * np.pi) * sigma_pri)) *
            np.exp(-0.5 * (1 / sigma_pri * (bins - mu_pri))**2))
            ax.plot(bins, y_pri, '-',color = 'black',label='prior')
    
            # add a 'best fit' line
            y = ((1 / (np.sqrt(2 * np.pi) * sigma[nv+istat,ist])) *
             np.exp(-0.5 * (1 / sigma[nv+istat,ist] * (bins - mu[nv+istat,ist]))**2))
            ax.plot(bins, y, '-', color= 'red', label='post')
    
            ax.set_title(r'#%s $\mu=$%.3f, $\sigma=$%.3f)'%(ist+1, mu[nv+istat,ist],sigma[nv+istat,ist]), fontsize=18)
            ax.tick_params(axis='x',labelsize=15)
            ax.tick_params(axis='y',labelsize=15)
            h1,l1 = ax.get_legend_handles_labels()
