#=================================================================================================
##################################################################################################
#                                              TODO:                                              
#
# 1. Include: min_mass method in the SCTarget class
#
##################################################################################################
#=================================================================================================
#
#     Project: DM-e Direct Detection & Astrophysics
#
#     Desciption : Semiconductors module
#                  A module that outputs different properties of semiconductor materials.
#
#     Tips : Where to start to use this module:
#
#            # to import this module:
#            >>>import semiconductors as sc
#            # to display this information:
#            >>>help(sc)
#            # to see the component keys of this dictionary (the instances of SCTarget included):
#            >>>print(sc.semiconductor_targets.keys())
#            # You can also call these instances by name:
#            >>>sc.the_target_you_want
#            # to see their attributes:
#            >>>vars(sc.semiconductor_targets['the_target_you_want'])
#            # or:
#            >>>vars(sc.the_target_you_want)
#
#
#     Author : Manuel A. Buen-Abad (February 2020)
#
#=================================================================================================

from __future__ import division # for division operator
import numpy as np

#---------------------------------------------------------------------------------------------------

# CONSTANTS AND CONVERSION FACTORS

amu2kg = 1.660538782e-27 # converts atomic mass unit [amu] to [kg]
kms2cms = 1.e5 # convert [km/s] to [cm/s]
a_em = 1/137. # fine structure constant
meeV = 0.511e6 # electron mass [eV]
ccms = 2.99792458e10 # speed of light [cm/s]

# NUMERICAL PARAMETERS
fileDirectory = "./semiconductors_data/essig_crystal_form_factors/"

qunit = 0.02*a_em*meeV # eV
Eunit = 0.1 # eV
k_mesh = 137 # see below Eq. (4.2) in 1509.01598 Essig et al. TODO: CHECK: DISCREPANCY: MATHEMATICA NOTEBOOK HAS 137, PAPER HAS 243. WHICH IS THE CORRECT VALUE USED IN THEIR CODE?
wk = 2./k_mesh # according to the convention from Quantum Espresso (see below Eq. (4.2) in 1509.01598 Essig et al. )

q_arr = qunit*np.linspace(1., 900., 900) - qunit/2. # momentum transfer [eV] array. It's the q-columns in 1509.01598 Essig et al.'s table of fcrystal2. sc.Si_num used by default, but it's the same with sc.Ge_num. TODO: The last term guarantees that each entry in the array is centered at the half-values.
E_arr = Eunit*np.linspace(1., 500., 500) - Eunit/2. # energy [eV] array. It's the E-rows in 1509.01598 Essig et al.'s table of fcrystal2. sc.Si_num used by default, but it's the same with sc.Ge_num. TODO: The last term guarantees that each entry in the array is centered at the half-values.
EGr, qGr = np.meshgrid(E_arr, q_arr, indexing='ij')

#---------------------------------------------------------------------------------------------------

# CLASSES

class SCTarget:
    """
    Class for semiconductor targets.
    """
    def __init__(self, material, Egap, epsilon, MCell, crystal_function='analytic', reference=None, Eprefactor=None, path=None):
        """
        An instance of SCTarget.
        
        Attributes
        ----------
        material : the semiconductor material
        Egap : the band-gap energy [eV]
        epsilon : mean energy per electron-hole pair produced by initial scattering [eV]
        MCell : the mass of the material's cells [kg]
        crystal_function : the type of function we are working with (default: 'analytic')
        reference : reference from whence the parameter values come (default: None)
        Eprefactor : the energy prefactor multiplying the crystal function (default: None)
        path : the path to the files with the data (default: None)
        fmatrix2 : the |f_[ik,i'k',G']|^2 matrix of the form factor for excitation from valence level {ik} to conduction level {i'k'} (Eq. 3.16 in 1509.01598 Essig et al.)
        fcrystal2 : the square of crystal form factor of the target material, according to 'reference'
        fshape : the shape of the fcrystal2 table, only present for the numerical case (E-rows, q-columns)
        """
        self._material = material
        self._Egap = Egap
        self._epsilon = epsilon
        self._MCell = MCell
        self._crystal_function = crystal_function
        self._reference = reference
        self._Eprefactor = Eprefactor
        self._path = path
        
        if bool(path):
            
            table = []
            with open (path, 'r') as f:
                lines = f.readlines()
                for i in lines:
                    i = i.rstrip()
                    table.append(i)
            
            # tab_shape = [int(el) for el in (table[0].split())]
            # table.pop(0)
            # table = np.array([float(el) for el in table])
            # table = table.reshape(tab_shape[1], tab_shape[0])

            # table = np.transpose(table)
            tab_shape = table[0].split()
            tab_shape = (int(tab_shape[1]), int(tab_shape[0]))
            table.pop(0)
            table = np.array([float(el) for el in table])
            table = table.reshape(tab_shape)
            
            self._fmatrix2 = table # E-rows (500), q-columns (900)
            factors = (self._Eprefactor/qunit/Eunit)*(wk/2.)*(wk/2.)*(1./wk)*q_arr # some factors that transform fmatrix2 into fcrystal2
            self._fcrystal2 = factors*self._fmatrix2 # E-rows (500), q-columns (900)
            self._fshape = tab_shape # E-rows (500), q-columns (900)
        
        else:
            AttributeError("WORK IN PROGRESS:\nSo far we can only give the crystal form factor from a table. Please give a valid \'path\' that stores the relevant data!")
    
    
    def Eth(self, Qth):
        """
        Threshold energy [eV] for a given ionization level.
        
        Parameters
        ----------
        Qth : ionization threshold
        """
        
        return (Qth-1)*self._epsilon + self._Egap
    
    
    
    def Qlvl(self, En):
        """
        The ionization level for a given deposited energy.
        
        Parameters
        ----------
        En : deposited energy [eV]
        """
        
        return int((En-self._Egap)/self._epsilon + 1.)
    
    
    
    def QbinE(self, Q, dE=Eunit):
        """
        The array of energies [eV] that correspond to a given bin of ionization level.
        Parameters
        ----------
        Q : the bin of ionization level
        dE : the steps of increment in energy [eV] (default: Eunit)
        """
        Elow = self.Eth(Q) # lowest end of the energy of the Q bin; inclusive
        Ehigh = min(self.Eth(Q+0.99), Eunit*target._fshape[0]) # highest end of the energy of the Q bin; inclusive (since this is np.arange), with a bit extra below
        
        return np.arange(Elow, Ehigh, dE)






Si_num = SCTarget(
                    'Si_num',
                    1.11,
                    3.6,
                    2*28.0855*amu2kg,
                    crystal_function='numeric',
                    reference='1509.01598 Essig et al.',
                    Eprefactor=2.0,
                    path=fileDirectory+'Si_f2.dat'
                )
Ge_num = SCTarget(
                    'Ge_num',
                    0.67,
                    2.9,
                    2*72.64*amu2kg,
                    crystal_function='numeric',
                    reference='1509.01598 Essig et al.',
                    Eprefactor=1.8,
                    path=fileDirectory+'Ge_f2.dat'
                )

# The dictionary of all the semiconductor targets under consideration
semiconductor_targets = {
                            'Si_num':Si_num,
                            'Ge_num':Ge_num
                        }
