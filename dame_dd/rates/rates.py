#=================================================================================================
##################################################################################################
#                                              TODO:                                              
#
# 1. Resolve: Is the q-integrand that of Essig's Eq. 3.13 or a mix of 3.13 and 4.4? (correction_option==0,1)
# 2. Resolve: Where are q_arr & E_arr: edges of units, or middles? (see last term of lines 61, 62)
# 3. Resolve: (Related): Do we bin with 'floor' or 'round'? (bin_option=='floor', 'round')
# 4. Resolve: Do we sum over array entries or do we use simps integration? (integration=='sum','simps')
# 5. Resolve: In binning, do I need to average over bin width?
#
##################################################################################################
#=================================================================================================
#=================================================================================================
#
#     Project: DM-e Direct Detection & Astrophysics
#
#     Desciption : Rates module
#                  A module that outputs different scattering rates, as a function of deposited
#                  energy or time of the year.
#
#     Tips : Where to start to use this module:
#
#            # to import this module:
#            >>>import rates as rt
#            # to display this information:
#            >>>help(rate)
#
#
#     Author : Manuel A. Buen-Abad (February 2020)
#
#=================================================================================================

from __future__ import division # for division operator
import numpy as np
import math
from math import pi
from itertools import cycle

from scipy.integrate import simps

from ..astrophysics import astrophysics as ap
from ..semiconductors import semiconductors as sc

#---------------------------------------------------------------------------------------------------

# CONSTANTS AND CONVERSION FACTORS
sec2year = 60*60*24*365.25 # converts [year] to [s]
kms2cms = 1.e5 # convert [km/s] to [cm/s]
a_em = 1/137. # fine structure constant
meeV = 0.511e6 # electron mass [eV]
ccms = 2.99792458e10 # speed of light [cm/s]
rho_chi = 0.4e9 #local DM density [eV/cm^3]


# NUMERICAL PARAMETERS
script_dir = str(__file__).rstrip('rates.py')
fileDirectory = "./rates_data/"

qunit = 0.02*a_em*meeV # eV
Eunit = 0.1 # eV
k_mesh = 137 # see below Eq. (4.2) in 1509.01598 Essig et al.
wk = 2./k_mesh # according to the convention from Quantum Espresso (see below Eq. (4.2) in 1509.01598 Essig et al. )

# TODO: generalize to other cases? what if atomic targets are included?
# q_arr = qunit*np.linspace(1., float(sc.Si_num._fshape[1]), sc.Si_num._fshape[1]) - qunit/2. # momentum transfer [eV] array. It's the q-columns in 1509.01598 Essig et al.'s table of fcrystal2. sc.Si_num used by default, but it's the same with sc.Ge_num. TODO: The last term guarantees that each entry in the array is centered at the half-values.
# E_arr = Eunit*np.linspace(1., float(sc.Si_num._fshape[0]), sc.Si_num._fshape[0]) - Eunit/2. # energy [eV] array. It's the E-rows in 1509.01598 Essig et al.'s table of fcrystal2. sc.Si_num used by default, but it's the same with sc.Ge_num. TODO: The last term guarantees that each entry in the array is centered at the half-values.
EGr, qGr = np.meshgrid(sc.E_arr, sc.q_arr, indexing='ij')

# FUNCTIONS DICTIONARY
components = ap.dark_matter_components.keys() # all the DM components defined in the astrophysics module

fn_dir = ap.init_fns(components=components, methods=['gvmin']) # defining the dictionary of functions with default parameters
# fn_dir = init_fns(components=components, methods=[], vArr=, timeArr=, vboost=, vsun=, t1=, from_scratch=, taken_from=, arrays=) # with other parameters

#---------------------------------------------------------------------------------------------------

# UNIVERSAL FUNCTIONS

#---------------------------------------------------------------------------------------------------
# Useful Functions
def my_floor(val, epsilon=1.e-6):
    """
    Floor function, that makes sure periodic .999... in floats are treated correctly.
    
    Parameters
    ----------
    val : value whose floor function we want to calculate
    epsilon : thershold of how close does a float with trailing periodic .999... needs to be to be considered the same as its rounded value (default: 1.e-6)
    """
    
    val2 = np.floor(val)
    
    if abs(val-(val2+1)) < epsilon:
        return np.int(val2) + 1
    else:
        return np.int(val2)



def my_round(val, eN=1.e6):
    """
    A rounding function, that makes sure periodic .4999... in floats are treated correctly.
    
    Parameters
    ----------
    val : value we want to round.
    eN : a large number we use to lift up the trailing .4999 above the decimal point, and then round up (default: 1.e6)
    """
    
    return int(np.round(np.round(val*eN)/eN))



def energy_bin(En, bin_option='floor'):
    """
    Finds the bin that corresponds to an energy Ed in the sc.E_arr. Note that bin = index+1.
    
    Parameters
    ----------
    En : energy [eV] for which we want to find its corresponding bin in sc.E_arr.
    bin_option : whether we are using the 'floor' or 'round' functions to obtain the bin (default: 'floor')
    """
    if bin_option == 'floor':
        return my_floor(En/sc.Eunit) + 1 # floor function to select the bin
    
    elif bin_option == 'round':
        return my_round(En/sc.Eunit) # rounds up or down to select the bin



#---------------------------------------------------------------------------------------------------
# Basic Functions
def mu_chie(m_chi):
    """
    DM-electron reduced mass [eV].
    
    Parameters
    ----------
    m_chi : DM mass [eV]
    """
    
    return meeV*m_chi/(meeV + m_chi)



def FDM(q, n):
    """
    DM form factor.
    
    Parameters
    ----------
    q : momentum transfer [eV]
    n : power
    """

    return (a_em*meeV/q)**n



def vmin(q, Ee, m_chi):
    """
    Minimum speed [km/s] a DM must have for the electron to gain the energy required. Comes from energy conservation.
    
    Parameters
    ----------
    q : momentum transfer [eV]
    Ee :  deposited energy Ee [eV]
    m_chi : DM mass [eV]
    """
    
    return ccms*(q/(2.*m_chi) + Ee/q)/kms2cms



#---------------------------------------------------------------------------------------------------

# RATES

#---------------------------------------------------------------------------------------------------

# Differential rate for fixed deposited energy and day of the year
def dRdE(m_chi, xsec, n, Ed, target_type='semiconductor', target_material='Si_num', MTarget=1., Time=1., dm_comp='SHM_default', variables='E', day=ap.tvernal, bin_option='floor', correction_option=0, integrate='sum', wanna_print=False):
    """
    DM-e scattering event rate per target mass per unit energy [# events/eV/kg/year]
    
    Parameters
    ----------
    m_chi : DM mass [eV]
    xsec : DM-e cross section [cm^2]
    n : power of DM form factor FDM(q,n)
    Ed : deposited energy [eV]
    target_type : whether 'semiconductor' or 'atomic' (default: 'semiconductor')
    target_material : the material that makes up the target (default: 'Si_num')
    MTarget : the mass of the target [kg] (default: 1 kg)
    Time : the unit time for the rate [yr] (default: 1 yr)
    dm_comp : the DM component under consideration (default: 'SHM_default')
    variables : whether 'E' (Energy) [eV] or 'Et'/'tE' (Energy [eV]-time [days]) (default: 'E')
    day : day of the year at which rate is measured (default: ap.tvernal, the vernal equinox)
    TODO: bin_option, correction_option, integrate, wanna_print
    """
    
    exposure = Time*sec2year*MTarget # the exposure in kg*s
    
    if target_type == 'semiconductor':
        
        if target_material not in sc.semiconductor_targets.keys(): raise ValueError("The \'target_material\' passed has not been instantiated in the \'semiconductors\' module. Either limit yourself to "+str(sc.semiconductor_targets.keys())+" or instantiate it in the \'semiconductors\' module and add it to the dictionary \'semiconductor_targets\'.")
        
        target = sc.semiconductor_targets[target_material] # the class of the semiconductor target material
        
        estimate = (exposure/target._MCell) * (rho_chi/m_chi) * xsec * (a_em*ccms) # a rough estimate of the logarithmic rate dR/dlnE (~ Time * N_Cell * <n_DM * xsec * v_rel>), which turns out to be also the first factor in the exact expression (see. Eq. 3.13 in Essig et al.)
        correction = 1. # some default value. Will be calculated below.
        
        if target._crystal_function == 'numeric':
            
            # Ei: the row index in the crystal function corresponding to the given value of the deposited energy.
            # Ei = int(math.floor(Ed/sc.Eunit)) - 1 
            Ei = energy_bin(Ed, bin_option=bin_option) - 1
            
            vmin_arr = vmin(sc.q_arr, Ed, m_chi) # shape: (len(sc.q_arr), ): that many q-columns
            
            if variables == 'E':
                # gvmin_vec = np.vectorize(fn_dir[dm_comp, 'gvmin', 'v']) # vectorizing gvmin [s/km] for the DM component considered
                # eta = gvmin_vec(vmin(sc.q_arr, Ed, m_chi))/kms2cms # same as gvmin, but in [s/cm]
                gvmin = fn_dir[dm_comp, 'gvmin', 'v']
                
                eta = gvmin(vmin_arr)/kms2cms # shape: (len(sc.q_arr), )
                
                del gvmin
            
            elif (variables == 'Et' or variables == 'tE'):
                # gvmin_vec = np.vectorize(fn_dir[dm_comp, 'gvmin', 'tv']) # vectorizing gvmin [s/km] for the DM component considered
                # eta = gvmin_vec(day, vmin(sc.q_arr, Ed, m_chi))/kms2cms # same as gvmin, but in [s/cm]
                gvmin = fn_dir[dm_comp, 'gvmin', 'vt']
                
                eta = gvmin(vmin_arr, day)/kms2cms # shape: (len(sc.q_arr), ): that many q-columns
                
                del gvmin
            
            else: raise ValueError("Parameter \'variables\' can only be either \'E\' or \'Et\'.")
            
            
            if wanna_print: print 'shape of vmin_arr:', vmin_arr.shape
            
            del vmin_arr            
            
            # correction: the other factor in the calculation of the rate
            # correction = sc.Eunit * (meeV**2. / mu_chie(m_chi)**2.) * sum( [ sc.qunit /(qi*sc.qunit) * (eta(qi)*ccms) * FDM(qi*sc.qunit, n)**2. * target._Eprefactor * 1./sc.qunit * 1./sc.Eunit * wk/2. * wk/2. * 1./wk * target._fcrystal2[qi-1, Ej-1] for qi in range(1, target._fshape[0]+1) ] ) # with eta as an exact function of continuous energy Ed
            if correction_option == 0:
                f2 = target._fcrystal2
            elif correction_option == 1:
                f2 = target._fmatrix2
            else:
                raise ValueError("Did not understand value of \'correction_option\'. Must be either 0 or 1.")
            
            if integrate == 'sum':
                correction = (meeV**2. / mu_chie(m_chi)**2.) * np.sum(sc.qunit * sc.q_arr**-2. * (eta*ccms) * FDM(sc.q_arr, n)**2. * f2[Ei], axis=-1) # np.sum(_, axis=-1): summing over the q-columns
            elif integrate == 'simps':
                correction = (meeV**2. / mu_chie(m_chi)**2.) * simps(sc.q_arr**-2. * (eta*ccms) * FDM(sc.q_arr, n)**2. * f2[Ei], sc.q_arr, axis=-1) # simps(_, axis=-1): integrating over the q-columns
            else: raise ValueError("Did not understand value of \'integrate\'. Must be either \'sum\' or \'simps\'.")
            
            if wanna_print: print "Ei="+str(Ei)+" [index_option="+bin_option+"],\tcorrection="+str(correction)+" [correction_option="+str(correction_option)+"]"
        
        else: raise AttributeError("WORK IN PROGRESS:\nSo far we can only give the crystal form factor numerically, from a table. Please select a value for \'target_material\' whose attribute \'_crystal_function\' is \'numeric\'.")
        
        return estimate*correction
    
    elif target_type == 'atomic':
        raise ValueError("!!!!!!!!!!!!!!!!!!\nMODULE UNDER CONSTRUCTION!\n!!!!!!!!!!!!!!!!!!\n")
    
    else:
        raise ValueError("\'target_type\' must be either \'semiconductor\' or \'atomic\'.")



# Differential energy spectrum rate
def dRdE_discrete(m_chi, xsec, n, target_type='semiconductor', target_material='Si_num', MTarget=1., Time=1., dm_comp='SHM_default', variables='E', day=ap.tvernal, tArr=ap.time_arr, EArr=sc.E_arr, qArr=sc.q_arr, correction_option=0, integrate='sum', wanna_print=False):
    """
    DM-e scattering event rate per target mass per unit energy [# events/eV/kg/year], evaluated over an E array or a t-E array.
    
    Parameters
    ----------
    m_chi : DM mass [eV]
    xsec : DM-e cross section [cm^2]
    n : power of DM form factor FDM(q,n)
    target_type : whether 'semiconductor' or 'atomic' (default: 'semiconductor')
    target_material : the material that makes up the target (default: 'Si_num')
    MTarget : the mass of the target [kg] (default: 1 kg)
    Time : the unit time for the rate [yr] (default: 1 yr)
    ingredients : the dictionary of DM components and their fractions (default: {'SHM_default':1.})
    variables : whether 'E' (Energy) [eV] or 'Et'/'tE' (Energy [eV]-time [days]) (default: 'E')
    tArr : time [days] array (default: ap.time_arr)
    EArr : energy [eV] array (default: sc.E_arr)
    qArr : momentum transfer [eV] array (default: sc.q_arr)
    TODO: correction_option, integrate, wanna_print
    """
    
    exposure = Time*sec2year*MTarget # the exposure in kg*s
    
    if target_type == 'semiconductor':
        
        if target_material not in sc.semiconductor_targets.keys(): raise ValueError("The \'target_material\' passed has not been instantiated in the \'semiconductors\' module. Either limit yourself to "+str(sc.semiconductor_targets.keys())+" or instantiate it in the \'semiconductors\' module and add it to the dictionary \'semiconductor_targets\'.")
        
        target = sc.semiconductor_targets[target_material] # the class of the semiconductor target material
        
        estimate = (exposure/target._MCell) * (rho_chi/m_chi) * xsec * (a_em*ccms) # a rough estimate of the logarithmic rate dR/dlnE (~ Time * N_Cell * <n_DM * xsec * v_rel>), which turns out to be also the first factor in the exact expression (see. Eq. 3.13 in Essig et al.)
        correction = 1. # some default value. Will be calculated below.
        
        if target._crystal_function == 'numeric':
            
            vminGr = vmin(qGr, EGr, m_chi)
            
            if variables == 'E':
                # gvmin_vec = np.vectorize(fn_dir[dm_comp, 'gvmin', 'v']) # vectorizing gvmin [s/km] for the DM component considered
                # eta = gvmin_vec(vminGr)/kms2cms # same as gvmin, but in [s/cm]
                gvmin = fn_dir[dm_comp, 'gvmin', 'v']
                eta = gvmin(vminGr)/kms2cms
                
                del gvmin
            
            elif (variables == 'Et' or variables == 'tE'):
                # gvmin_vec = np.vectorize(fn_dir[dm_comp, 'gvmin', 'tv']) # vectorizing gvmin [s/km] for the DM component considered
                # eta = gvmin_vec(day, vminGr)/kms2cms # same as gvmin, but in [s/cm]
                gvmin = fn_dir[dm_comp, 'gvmin', 'vt']
                eta = np.array([gvmin(vminGr[:,i], tArr).T for i in range(len(sc.q_arr))]).T / kms2cms
                
                del gvmin
            
            else: raise ValueError("Parameter \'variables\' can only be either \'E\' or \'Et\'.")
            
            if wanna_print: print 'shape of vminGr:', vminGr.shape
            
            del vminGr
            
            if correction_option == 0:
                f2 = target._fcrystal2
            elif correction_option == 1:
                f2 = target._fmatrix2
            else:
                raise ValueError("Did not understand value of \'correction_option\'. Must be either 0 or 1.")
            
            if integrate == 'sum':
                correction = (meeV**2. / mu_chie(m_chi)**2.) * np.sum(sc.qunit * sc.q_arr**-2. * (eta*ccms) * FDM(sc.q_arr, n)**2. * f2, axis=-1) # np.sum(_, axis=-1): summing over the q-columns
            elif integrate == 'simps':
                correction = (meeV**2. / mu_chie(m_chi)**2.) * simps(sc.q_arr**-2. * (eta*ccms) * FDM(sc.q_arr, n)**2. * f2, sc.q_arr, axis=-1) # simps(_, axis=-1): integrating over the q-columns
            else: raise ValueError("Did not understand value of \'integrate\'. Must be either \'sum\' or \'simps\'.")
            
            if wanna_print: print 'shape of correction:', correction.shape
        
        else: raise AttributeError("WORK IN PROGRESS:\nSo far we can only give the crystal form factor numerically, from a table. Please select a value for \'target_material\' whose attribute \'_crystal_function\' is \'numeric\'.")
        
        return estimate*correction
    
    elif target_type == 'atomic':
        raise ValueError("!!!!!!!!!!!!!!!!!!\nMODULE UNDER CONSTRUCTION!\n!!!!!!!!!!!!!!!!!!\n")
    
    else:
        raise ValueError("\'target_type\' must be either \'semiconductor\' or \'atomic\'.")



def binning(spectrum, Eini=0., dE=1., bin_option='floor', integrate='sum', wanna_print=False):
    """
    Binning of a given spectrum [# events/eV/kg/year].
    
    Parameters
    ----------
    spectrum : DM-e scattering event rate spectrum [# events/eV/kg/year] as a function of energy [eV].
    Eini : left-most/initial value of the energy [eV] bins (default: 0.)
    dE : energy [eV] bin widths (default: 1.)
    
    TODO: bin_option, integrate, wanna_print
    """
    idx_E0 = energy_bin(Eini, bin_option=bin_option)-1 # the index in sc.E_arr that corresponds to the initial energy Eini
    reduced_Earr = sc.E_arr[idx_E0:] # the reduced array of energies, dropping everything to the left of Eini
    reduced_spectrum = spectrum[idx_E0:] # the reduced spectrum, dropping everything to the left of Eini
    
    NEs = my_floor(dE/sc.Eunit) # the number of indices from sc.E_arr that are in an energy bin
    
    ELs = (reduced_Earr[::NEs])[:-1] - sc.Eunit/2. # the energies from reduced_Earr corresponding to the left edges of the energy bins, but the last one (to avoid possible cases where the last bin has a width smaller than NEs)
    ECs = ELs + NEs*sc.Eunit/2. # the energies from reduced_Earr corresponding to the centers of the energy bins
    
    if wanna_print: print 'idx_E0=', idx_E0, '\n', 'len(reduced_E_arr)=', len(reduced_Earr), '\n', 'len(reduced_spectrum)=', len(reduced_spectrum), '\n', 'NEs=', NEs, '\n', 'ELs=', ELs, '\n', 'ECs=', ECs, '\n', 
    
    if integrate == 'sum':
        binned_rates = np.array([ np.sum(reduced_spectrum[i*NEs:(i+1)*NEs]) / NEs for i in range(len(ECs)) ])
    elif integrate == 'simps':
        binned_rates = np.array([ simps(reduced_spectrum[i*NEs:(i+1)*NEs], reduced_Earr[i*NEs:(i+1)*NEs]) / (NEs*sc.Eunit) for i in range(len(ECs)) ])
    else: raise ValueError("Did not understand value of \'integrate\'. Must be either \'sum\' or \'simps\'.")
    
    if wanna_print: print 'NEs=', NEs, '\n', 'len(reduced_spectrum[NEs:2*NEs])=', len(reduced_spectrum[NEs:2*NEs])
    
    return ECs, binned_rates



def Qbinning(spectrum, target_material, bin_option='floor', integrate='sum', wanna_print=False):
    """
    Binning of a given spectrum according to the ionization level of the semiconductor.
    
    Parameters
    ----------
    """
    
    target = sc.semiconductor_targets[target_material] # the class of the semiconductor target material
    Eini = target._Egap
    dE = target._epsilon
    
    Es, rts = binning(spectrum, Eini=Eini, dE=dE, bin_option=bin_option, integrate=integrate, wanna_print=wanna_print)
    rts *= dE # we want the total rate in each Q-bin, i.e. the number of events, not the rate spectrum
    
    Qlist = np.vectorize(target.Qlvl)
    
    Qs = Qlist(Es)
    
    if wanna_print: print 'Es=', Es, '\n', 'Qs=', Qs, '\n', 'rts=', rts
    
    return Qs, rts



def Tbinning(tQspectrum, ntbins=12, years=1, rescaled_exposure=True, total_rate=False, integrate=True, Nti=20, wanna_print=False):
    """
    Function that takes a Q-spectrum continuous in time bins it in the time axis.
    
    Parameters
    ----------
    tQspectrum : the continuous Q-spectrum for 1 year
    ntbins : number of time bins in a single year (default: 12)
    years : number of years (default: 1)
    rescaled_exposure : whether the events will be rescaled by the time-bin exposure (default: True)
    total_rate : whether we want to sum over the Q-bins (default: False)
    integrate : whether we want to use integration for the binning (default: True)
    Nti : number of time points per time bin for binning via integration (default: 10)
    wanna_print : whether we want to print at different checkpoints (default: False)
    """
    
    new_spectrum = [tQspectrum]*years
    new_spectrum = np.concatenate(new_spectrum)
    new_spectrum = new_spectrum.T# Q-rows and t-columns
    ns_shape = new_spectrum.shape# Q-rows and t-columns
    full_time_arr = np.linspace(0, 365.25*years, ns_shape[1])# the time array
    
    if wanna_print:
        print ns_shape
    
    tbinned = []
    if (((ns_shape[1]-1) % ntbins*years) == 0) and (not integrate):# not gonna integrate: gonna sum instead
        
        if wanna_print:
            print 'summing!'
        
        Nti = int((ns_shape[1]-1)/(ntbins*years))
        
        for Qi, osc in enumerate(new_spectrum):
            
            fixed_Q = []
            
            for i in xrange(ntbins*years):
                fixed_Q.append(sum(osc[i*Nti:(i+1)*Nti])/float(Nti))
            
            tbinned.append(fixed_Q)
    
    else:
        
        if wanna_print:
            print 'integrating!'
        
        for Qi, osc in enumerate(new_spectrum):
            osc_fn = interp1d(full_time_arr, osc, kind='linear', fill_value='extrapolate')
            
            fixed_Q = []
            
            for i in xrange(ntbins*years):
                time_arr = np.linspace(i*365.25/ntbins, (i+1)*365.25/ntbins, Nti)
                value = simps(osc_fn(time_arr), time_arr)/(365.25/ntbins)
                fixed_Q.append(value)
            tbinned.append(fixed_Q)
    
    tbinned = np.array(tbinned)
    tbinned = tbinned.T
    
    if rescaled_exposure:
        tbinned /= (ntbins*years)
    if total_rate:
        tbinned = np.sum(tbinned, axis=-1)
    
    return tbinned





#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
#
#
#                      S E C T I O N       U N D E R       D E V E L O P M E N T
#
#
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
