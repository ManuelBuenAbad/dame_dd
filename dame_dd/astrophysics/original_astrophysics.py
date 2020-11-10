#=================================================================================================
##################################################################################################
#                                              TODO:                                              
#
# 1. Pythonize: Combine DMComponent and DMVelDistr (N_esc & other subtleties)
# 2. Improve: Allow export and import from 2D data, by allowing different lengths of the keyword parameter 'arrays'
#
##################################################################################################
#=================================================================================================
#=================================================================================================
#
#     Project: DM-e Direct Detection & Astrophysics
#
#     Desciption : Astrophysics module
#                  A module that outputs different moments of the astrophysical DM velocity distributions,
#                  defined internally here.
#
#     Tips : Where to start to use this module:
#
#            # to import this module:
#            >>>import astrophysics as ap
#            # to display this information:
#            >>>help(ap)
#            # to see the component keys of this dictionary (the instances of DMComponent included):
#            >>>print(ap.dark_matter_components.keys())
#            # You can also call these instances by name:
#            >>>ap.the_component_key_you_want
#            # to see their attributes:
#            >>>vars(ap.dark_matter_components['the_component_key_you_want'])
#            # or:
#            >>>vars(ap.the_component_key_you_want)
#
#
#     Author : Manuel A. Buen-Abad (January-February 2020)
#
#=================================================================================================

from __future__ import division # for division operator
import numpy as np
from numpy.linalg import norm
from math import pi

from scipy.special import erf
from scipy.integrate import simps
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline

#---------------------------------------------------------------------------------------------------

# ASTROPHYSICAL PARAMETERS:

# Galactic Rest Frame (GRF)
# @ 1901.02016
vEsc = 528. # escape velocity [km/s]
dVEsc = 24.5 # 1-sigma error [km/s]

# SHM DM dispersion
# @ 1807.02519 & 1909.04684
v0 = 235. # SHM DM mean velocity [km/s]
dv0 = 3. # 1-sigma error [km/s]
vcGaia = 235. # local circular velocity [km/s]
dvcGaia = 3. # 1-sigma error [km/s]

# Local Standard of Rest (LSR) is the frame that follows the mean motion of material in the Milky Way in the neighborhood of the Sun

# @ 1608.00971 + 1904.05721
vLSR = np.array([0.0, 235.0, 0.0]) # velocity [km/s] of the LSR w.r.t. GRF
dvLSR = np.array([0.0, 3.0, 0.0]) # 1-sigma error [km/s]

# @ 0912.3693
vSun_pec = np.array([11.1, 12.24, 7.25]) # Sun's peculiar velocity [km/s] w.r.t. LSR frame
dvSun_pec = np.array([0.74, 0.47, 0.37]) # 1-sigma error [km/s]

v_sun = vLSR + vSun_pec # The velocity [km/s] of the Sun w.r.t. GRF
sun_hat = v_sun/norm(v_sun) # unit vector along the direction of the Sun's velocity w.r.t. GRF

vEarth_orbit = 29.79 # Earth's orbital speed around the Sun w.r.t. Sun's frame

# Directional unit vectors for Earth's motion w.r.t. Sun's frame.
eps0 = np.array([0.9940, 0.1095, 0.0031])
eps1 = np.array([-0.0517, 0.4945, -0.8677])
# Making sure these unit vectors are exactly normalized.
eps0 = eps0/norm(eps0)
eps1 = eps1/norm(eps1)
epsilon_params = (eps0, eps1) 

tvernal = 79.6 # in days: March 21 (vernal equinox)
year = 365.25 # in days


# NUMERICAL PARAMETERS
fileDirectory = "./astrophysics_data/"

# The number of points in the arrays for the different variables
v_pts = 1001 # increments of 1 km/s
th_pts = 51
a_pts = 51
b_pts = 51
t_pts = 105 # increments of ~0.5 weeks
# t_pts = 366 # increments of ~1 days

# The (1D) arrays to be used for integration and plotting
v_arr = np.linspace(0., 1000., v_pts) # [km/s]
theta_arr = np.linspace(0., pi, th_pts) # [radians]
alpha_arr = np.linspace(0., pi, a_pts) # [radians]
beta_arr = np.linspace(0., 2.*pi, b_pts) # [radians]
time_arr = np.linspace(0., year, t_pts) # [days]

# The grids
vel_2D_grid, theta_2D_grid = np.meshgrid(v_arr, theta_arr, indexing='ij') # v-theta 2D grid
alpha_2D_grid, beta_2D_grid = np.meshgrid(alpha_arr, beta_arr, indexing='ij') # alpha-beta 2D grid
vel_3D_grid, alpha_3D_grid, beta_3D_grid = np.meshgrid(v_arr, alpha_arr, beta_arr, indexing='ij') # v-alpha-beta 3D grid
time_2Dt_grid, vel_2Dt_grid = np.meshgrid(time_arr, v_arr, indexing='ij') # v-time 2D grid


#---------------------------------------------------------------------------------------------------

# UNIVERSAL FUNCTIONS

def v_earth(vsun=v_sun, t1=tvernal):
    """
    The Earth's velocity [km/s] w.r.t. GRF, as a function of days.
    
    Parameters
    ----------
    vsun : the velocity [km/s] of the Sun w.r.t. GRF (default: v_sun)
    t1 : an arbitrary initial reference time [days] (default: tvernal)
    """
    
    vearth = lambda t: vEarth_orbit*( np.array(map(lambda eps_i: eps_i*np.cos(2.*pi*(t-t1)/year), epsilon_params[0])) + np.array(map(lambda eps_i: eps_i*np.sin(2.*pi*(t-t1)/year), epsilon_params[1])) ) # this is a funny way to multiply epsilon_params[0]*cos() and epsilon_params[1]*sin(), necessary because sometimes we want cos()/sin() to be a number and sometimes to be an array/grid in the time dimension.
    
    vobs =  lambda t: np.array(map(lambda vsun_i, vearth_i: vsun_i+vearth_i, vsun, vearth(t))).T # this is a funny way to add vsun+vearth, necessary because sometimes vearth is a 3-dim velocity vector, but others it is an array (in the time dimension) of 3-dim velocity vectors.
    
    mag = lambda t: norm(vobs(t), axis=-1) # the axis=-1 parameter ensures that the norm is calculated regardless of whether vobs is a 3-dim vector of numbers or of arrays/grids in the time dimension.
    
    return vobs, mag



def modulation_params(vdm_wrt_grf, vsun=v_sun, t1=tvernal):
    """
    Returns a tuple:
    b : The parameter that relates the orthogonality of the DM's velocity in the Sun's frame w.r.t. the Earth's plane
    phase : The time of the year [days] at which the DM speed w.r.t. the Earth's frame is the largest
    counter_phase : The time of the year [days] at which the DM speed w.r.t. the Earth's frame is the smallest
    
    Parameters
    ----------
    vdm_wrt_grf : the DM velocity [km/s] w.r.t. GRF
    vsun : the velocity [km/s] of the Sun w.r.t. GRF (default: v_sun)
    t1 : an arbitrary initial reference time [days] (default: tvernal)
    """
    
    omega = 2.*pi / year # the angular frequency of the Earth's orbit
    phi1 = omega*t1 # the angular phase of the reference time t1
    
    vdm_wrt_sun = vdm_wrt_grf - vsun # DM velocity w.r.t. the Sun's frame
    
    b1 = np.inner(epsilon_params[0], vdm_wrt_sun) / norm(vdm_wrt_sun)
    b2 = np.inner(epsilon_params[1], vdm_wrt_sun) / norm(vdm_wrt_sun)
    
    b = np.sqrt(b1**2. + b2**2.)
    
    beta1 = b1/b
    beta2 = b2/b
    
    phase = ( ( phi1 + np.sign(-beta2)*np.arccos(-beta1) ) / omega ) % year # modulo 1 year, obtained from trigonometry; note the dependence on the sign of beta2
    counter_phase = (phase + year/2.) % year # modulo 1 year, obtained from trigonometry
    
    return b, phase, counter_phase



#---------------------------------------------------------------------------------------------------

# CLASSES

class DMComponent:
    """
    Class for Dark Matter components
    """
    def __init__(self, name, reference, frame, coordinates, mean, sigma, errors={'mean':np.zeros(3), 'sigma':np.zeros(3)}, truncated=True, path=None):
        """
        An instance of DMComponent.
        
        Attributes
        ----------
        name : name of the component
        reference : reference from whence the parameter values come
        frame : reference frame
        coordinates : (3, 'cartesian'), (3, 'cylindrical'), (1, 'spherical')
        mean : mean speed w.r.t. galactic rest frame [km/s]
        sigma : standard dispersion w.r.t. galactic rest frame [km/s]
        truncated : whether the distribution is truncated at the escape velocity (default: True)
        path : the path to the files with the data (default: None)
        
        """
        self._name = name
        self._reference = reference
        self._frame = frame
        self._coordinates = coordinates
        self._mean = mean
        self._sigma = sigma
        self._errors = errors
        self._truncated = truncated
        self._path = path
        
    @property
    def N_esc(self):
        """
        The normalization factor multiplying the distribution.
        """
        return self._N_esc
    
    @N_esc.setter
    def N_esc(self, value):
        """
        Sets the normalization to 'value'.
        """
        if bool(value):
            self._N_esc = value
        else:
            self._N_esc = 1.

    def vel_distr_3D(self, v, alpha, beta, vboost=v_sun, integrand=False, moment=0., vmin=0.):
        """
        The 3D velocity distribution function f=dN/dVv [(km/s)^-3], where dVv=v^2*dv*dOmega is the *velocity* volume element (i.e. it is not real space volume but velocity volume). Can also output this distribution times the integration measure and a vmin mask for different velocity moments.
        
        Parameters
        ----------
        v : the magnitude of the velocity [km/s]
        alpha : the polar angle (from the fixed zenith direction)
        beta : the azimuthal angle
        vboost : the boost velocity of the frame under consideration [km/s] (default: v_sun)
        integrand : whether this is an integrand or simply the 3D distribution (default: False)
        moment : if 'integrand'==True, the moment of the velocity to be considered (default: 0.)
        vmin : if 'integrand'==True, the value of the minimum speed [km/s] under consideration (default: 0.)
        """
        
        pref = (np.prod(self._sigma)**2. * (2.*pi)**3.)**-0.5 # the prefactor multiplying the distribution
        
        vx = v*np.sin(alpha)*np.cos(beta) # the velocity in the x-direction
        vy = v*np.sin(alpha)*np.sin(beta) # the velocity in the y-direction
        vz = v*np.cos(alpha) # the velocity in the z-direction
        
        vvec = np.array([vx, vy, vz]) # defining velocity vector
        del vx, vy, vz # some garbage collection
        
        vvec = np.array(map(lambda v_i, vb_i: v_i + vb_i, vvec, vboost)) # adding vboost to get total vector
        
        vvecmag = norm(vvec, axis=0) # calculating the norm of the total vector
        
        mask = np.heaviside(vEsc - vvecmag, 0.) if self._truncated else 1. # masking velocities larger than the escape velocity
        
        # calculating the gaussian factors:
        exp_yz = np.exp( -(vvec[1]-self._mean[1])**2. / (2.*self._sigma[1]**2.) ) * np.exp( -(vvec[2] - self._mean[2])**2. / (2.*self._sigma[2]**2.) ) # the y and z gaussian factors

        if self._name == 'Gaia_sausage':
            exp_x = 0.5*np.exp( -(vvec[0] - self._mean[0])**2. / (2.*self._sigma[0]**2.) ) + 0.5*np.exp( -(vvec[0] + self._mean[0])**2. / (2.*self._sigma[0]**2.) ) # the bi-modal x gaussian factor for the sausage
        
        else:
            exp_x = np.exp( -(vvec[0]-self._mean[0])**2. / (2.*self._sigma[0]**2.) ) # the x gaussian factor
        
        
        distribution = pref/self._N_esc *exp_x*exp_yz*mask # the final distribution

        del pref, vvec, vvecmag, mask, exp_yz, exp_x # some garbage collection
        
        if integrand: # if we want not only the distribution but the full integrand for a moment of velocity, we need extra factors
                
            measure = v**(2.+moment)*np.sin(alpha) # the measure of the integrand
            vmin_mask = np.heaviside(v-vmin, 1.) # masking velocities below vmin
            
        else:
            measure = 1.
            vmin_mask = 1.
        
        return distribution*measure*vmin_mask
    
    def vel_distr_2D(self, v, theta, vboost=v_sun, integrand=False, moment=0., vmin=0.):
        """
        The 2D velocity distribution function [(km/s)^-3], having integrated over the azimuthal angle. Useful when the distributions are isotropic in the galactic rest frame (like the SHM). This should be equal to 2pi*vel_distr_3D for these components. Can also output this distribution times the integration measure and a vmin mask for different velocity moments.
        
        Parameters
        ----------
        v : the magnitude of the velocity [km/s]
        theta : the polar angle (from the fixed zenith direction)
        vboost : the boost velocity of the frame under consideration [km/s] (default: v_sun)
        integrand : whether this is an integrand or simply the 3D distribution (default: False)
        moment : if 'integrand'==True, the moment of the velocity to be considered (default: 0.)
        vmin : if 'integrand'==True, the value of the minimum speed [km/s] under consideration (default: 0.)
        """
        
        pref = (np.prod(self._sigma)**2. * (2.*pi)**3.)**-0.5
        
        if self._coordinates[0] == 1:

            vMP = np.sqrt(2.)*self._sigma[0] # most probable velocity
            
            vboostmag = norm(vboost, axis=0) # the magnitude of vboost
            
            vtot = np.sqrt(v**2. + 2.*v*vboostmag*np.cos(theta) + vboostmag**2.) # magnitude of total vector
            
            mask = np.heaviside(vEsc - vtot, 0.) if self._truncated else 1. # masking velocities larger than the escape velocity
            
            distribution = 2.*pi*pref/self._N_esc *np.exp(-vtot**2. / vMP**2.)*mask # the final distribution

            del pref, vMP, vboostmag, vtot, mask # some garbage collection
            
            if integrand: # if we want to output the integrand, we need to multiply by the integration measure and a possible vmin mask.
                
                measure = v**(2.+moment)*np.sin(theta) # the measure of the integrand
                vmin_mask = np.heaviside(v-vmin, 1.) # masking velocities below vmin
                
            else:
                measure = 1.
                vmin_mask = 1.
            
            return distribution*measure*vmin_mask
        
        else:
            raise AttributeError("The vel_distr_2D method cannot be used for this instance of DMComponent, since its value for the self._coordinates[0] attribute is not 1: it is not an isotropic distribution!")
    
    def load_fv(self):
        """
        Loads the f(v) velocity distribution from a file located at self._path, and returns a tuple:
        v_bins : the velocity bins [km/s]
        f_v_arr : the array of dN/dv [(km/s)^-1]
        """
        
        if bool(self._path):
            
            v_bins, f_v_arr = np.loadtxt(self._path, unpack=True)
            
            return v_bins, f_v_arr
        else:
            raise AttributeError("The load_fv method cannot be used for this instance of DMComponent: there is not a valid path associated with it!")

class DMVelDistr(DMComponent):
    """
    Class for the DM velocity distributions, a subclass of DMComponent.
    """
    def integral(self, vboost=np.array([0., 0., 0.])):
        """
        The (normalized) DM density, i.e. the integral over the (velocity) volume of the velocity distribution function. Should therefore always be 1.
        
        Parameters
        ----------
        vboost : the boost velocity of the frame under consideration [km/s] (default: np.array([0., 0., 0.]))
        """
        if self._coordinates[0] == 1:
            
            return simps(simps(
               self.vel_distr_2D(vel_2D_grid, theta_2D_grid, vboost=vboost, integrand=True),
                theta_arr), v_arr)
        
        else:
            
            return simps(simps(simps(
                self.vel_distr_3D(vel_3D_grid, alpha_3D_grid, beta_3D_grid, vboost=vboost, integrand=True),
                beta_arr), alpha_arr), v_arr)
    
    def fv(self, v, vboost=v_sun, taken_from='3D', arrays=(alpha_arr, beta_arr)):
        """
        The (normalized) DM velocity distribution dN/dv [(km/s)^-1], having integrated out all angular dependence.
        
        Parameters
        ----------
        v : the magnitude of the velocity [km/s]
        vboost : the boost velocity of the frame under consideration [km/s] (default: v_sun)
        taken_from : whether the distribution is taken from an analytic '3D', '2D', or 'numeric' expression
        arrays : a tuple of the arrays for the different variables (default: (alpha_arr, beta_arr))
        """
        
        if taken_from == '3D':
            
            if len(arrays) != 2:
                raise ValueError("If \'taken_from\'=\'3D\' then we need two arrays for the two angular variables in 3D, i.e. len('arrays')==2")
            
            aArr, bArr = arrays
            aGr, bGr = np.meshgrid(aArr, bArr, indexing='ij')
            
            return simps(simps(
                self.vel_distr_3D(v, aGr, bGr, vboost=vboost, integrand=True),
                bArr), aArr)
            
        elif (taken_from == '2D') and (self._coordinates[0] == 1):
            
            if (type(arrays) == tuple and len(arrays) != 1):
                raise ValueError("If \'taken_from\'=\'2D\' then we need one array for the theta angular variable in 2D, i.e. \'arrays\' has to be a single array of values for theta.")
            
            thArr = arrays
            
            return simps(
                self.vel_distr_2D(v, thArr, vboost=vboost, integrand=True),
                thArr)
            
        elif (taken_from == '2D') and (self._coordinates[0] != 1):
            
            raise AttributeError("The fv method with \'taken_from\'=\'2D\' cannot be used for this instance of DMComponent, since its value for the self._coordinates[0] attribute is not 1: it is not an isotropic distribution!")
        
        elif (taken_from == 'numeric') and (bool(self._path) == True):
            
            vbins, fvs = self.load_fv()
            fv_num = interp1d(vbins, fvs, fill_value='extrapolate')
            
            return fv_num(v)
        
        elif (taken_from == 'numeric') and (bool(self._path) == False):
            
            raise AttributeError("The fv method with \'taken_from\'=\'numeric\' cannot be used for this instance of DMComponent: there is not a valid path associated with it!")
        
        else:
            
            raise ValueError("The fv method cannot be calculated with the value for \'taken_from\' passed. It must be either \'3D\', \'2D\', or \'numeric\'.")
        
    
    def moment_fv(self, vmin, moment, vboost=v_sun, taken_from='3D', arrays=(v_arr, alpha_arr, beta_arr)):
        """
        A given moment of the fv distribution [(km/s)^moment].
        
        Parameters
        ----------
        vmin : the minimum speed (in the lab frame) the DM must have in order to scatter [km/s]
        moment : the moment under consideration
        vboost : the boost velocity of the frame under consideration [km/s] (default: v_sun)
        taken_from : whether the distribution is taken from an analytic '3D', '2D', or 'numeric' expression
        arrays : a tuple of the arrays for the different variables (default: (v_arr, alpha_arr, beta_arr))
        """
        
        if taken_from == '3D':
            
            if len(arrays) != 3:
                raise ValueError("If \'taken_from\'=\'3D\' then we need three arrays for 3D variables, i.e. len('arrays')==3")
            
            vArr, aArr, bArr = arrays
            vGr, aGr, bGr = np.meshgrid(vArr, aArr, bArr, indexing='ij')
            
            return simps(simps(simps(
                self.vel_distr_3D(vGr, aGr, bGr, vboost=vboost, integrand=True, moment=moment, vmin=vmin),
                bArr), aArr), vArr)
            
        elif (taken_from == '2D') and (self._coordinates[0] == 1):
            
            if len(arrays) != 2:
                raise ValueError("If \'taken_from\'=\'2D\' then we need three arrays for 2D variables, i.e. len('arrays')==2")
            
            vArr, thArr = arrays
            vGr, thGr = np.meshgrid(vArr, thArr, indexing='ij')
            
            return simps(simps(
                self.vel_distr_2D(vGr, thGr, vboost=vboost, integrand=True, moment=moment, vmin=vmin),
                thArr), vArr)
            
        elif (taken_from == '2D') and (self._coordinates[0] != 1):
            
            raise AttributeError("The moment_fv method with \'taken_from\'=\'2D\' cannot be used for this instance of DMComponent, since its value for the self._coordinates[0] attribute is not 1: it is not an isotropic distribution!")
        
        elif (taken_from == 'numeric') and (bool(self._path) == True):
            
            vbins, fvs = self.load_fv()
            
            try:
                return simps(vbins**moment * fvs * np.heaviside(vbins-vmin), vbins)
            except ZeroDivisionError:
                return 0.
        
        elif (taken_from == 'numeric') and (bool(self._path) == False):
            
            raise AttributeError("The moment_fv method with \'taken_from\'=\'numeric\' cannot be used for this instance of DMComponent: there is not a valid path associated with it!")
        
        else:
            
            raise ValueError("The moment_fv method cannot be calculated with the value for \'taken_from\' passed. It must be either \'3D\', \'2D\', or \'numeric\'.")
    
    def gvmin(self, vmin, vboost=v_sun, taken_from='3D', arrays=(v_arr, alpha_arr, beta_arr)):
        """
        The inverse mean speed [s/km].
        
        Parameters
        ----------
        vmin : the minimum speed (in the lab frame) the DM must have in order to scatter [km/s]
        vboost : the boost velocity of the frame under consideration [km/s] (default: v_sun)
        taken_from : whether the distribution is taken from an analytic '3D' or '2D' expression
        arrays : a tuple of the arrays for the different variables (default: (v_arr, alpha_arr, beta_arr))
        """
        
        return self.moment_fv(vmin=vmin, moment=-1., vboost=vboost, taken_from=taken_from, arrays=arrays)
    
    def etamin(self, vmin, vboost=v_sun, taken_from='3D', arrays=(v_arr, alpha_arr, beta_arr)):
        """
        The inverse mean speed [s/km]. Same as gvmin.
        
        Parameters
        ----------
        vmin : the minimum speed (in the lab frame) the DM must have in order to scatter [km/s]
        vboost : the boost velocity of the frame under consideration [km/s] (default: v_sun)
        taken_from : whether the distribution is taken from an analytic '3D' or '2D' expression
        arrays : a tuple of the arrays for the different variables (default: (v_arr, alpha_arr, beta_arr))
        """
        
        return self.gvmin(vmin, vboost=vboost, taken_from=taken_from, arrays=arrays)
    
    def hmin(self, vmin, vboost=v_sun, taken_from='3D', arrays=(v_arr, alpha_arr, beta_arr)):
        """
        The mean speed [km/s].
        
        Parameters
        ----------
        vmin : the minimum speed (in the lab frame) the DM must have in order to scatter [km/s]
        vboost : the boost velocity of the frame under consideration [km/s] (default: v_sun)
        taken_from : whether the distribution is taken from an analytic '3D' or '2D' expression
        arrays : a tuple of the arrays for the different variables (default: (v_arr, alpha_arr, beta_arr))
        """
        
        return self.moment_fv(vmin=vmin, moment=1., vboost=vboost, taken_from=taken_from, arrays=arrays)


################################################################################
# Instances of DMVelDistr:



SHM_default = DMVelDistr(
                'SHM_default',
                '1807.02519 & 1909.04684',
                'galacto-centric',
                (1, 'spherical'),
                np.zeros(3),
                v0/np.sqrt(2.)*np.ones(3),
                errors={'mean':(np.zeros(3), -np.zeros(3)), 'sigma':(dv0/np.sqrt(2.)*np.ones(3) -dv0/np.sqrt(2.)*np.ones(3))},
                truncated=True,
                path=None
                )
SHM_default._N_esc = (erf(vEsc/v0) - 2.*vEsc/np.sqrt(pi)/v0 * np.exp(-(vEsc/v0)**2.))
SHM_default._N_esc *= 0.9998944463084201



SHM_infinity = DMVelDistr(
                'SHM_infinity',
                '1807.02519 & 1909.04684',
                'galacto-centric',
                (1, 'spherical'),
                np.zeros(3),
                v0/np.sqrt(2.)*np.ones(3),
                errors={'mean':(np.zeros(3), -np.zeros(3)), 'sigma':(dv0/np.sqrt(2.)*np.ones(3) -dv0/np.sqrt(2.)*np.ones(3))},
                truncated=False,
                path=None
                )
SHM_infinity._N_esc = 1.00000001920081



Gaia_halo = DMVelDistr(
                'Gaia_halo',
                '1807.02519 Necib et al. & files of sampled distributions',
                'galacto-centric',
                (3, 'spherical'),
                np.zeros(3),
                np.array([143.96, 132.03, 118.30]),
                errors={'mean':(np.zeros(3), -np.zeros(3)), 'sigma':(np.array([4.31, 4.30, 3.42]), np.array([-5.03, -3.57, -1.86]))},
                truncated=True,
                path=fileDirectory+"DM_Velocity_Distribution-master/f_v_halo_normalized.txt"
                )
Gaia_halo._N_esc = 0.9986426837618283



Gaia_halo_median = DMVelDistr(
                'Gaia_halo_median',
                '1807.02519 Necib et al.',
                'galacto-centric',
                (3, 'spherical'),
                np.array([0., 0., 0.]),
                np.array([140.3, 125.9, 114.2]),
                errors={'mean':(np.zeros(3), -np.zeros(3)), 'sigma':(np.array([+4.2, +4.1, +3.3]), np.array([-4.9, -3.4, -1.8]))},
                truncated=True,
                path=fileDirectory+"DM_Velocity_Distribution-master/f_v_halo_normalized.txt"
                )
Gaia_halo_median._N_esc = 0.999170851606421



Gaia_sausage = DMVelDistr(
                'Gaia_sausage',
                '1807.02519 Necib et al. & files of sampled distributions',
                'galacto-centric',
                (3, 'spherical'),
                np.array([115.50, 36.94, -2.92]),
                np.array([108.33, 62.60, 57.99]),
                errors={'mean':(np.array([1.77, 1.87, 0.85]), np.array([-2.06, -1.87, -0.85])), 'sigma':(np.array([1.20, 1.53, 0.70]), np.array([-1.30, -1.53, -0.80]))},
                truncated=True,
                path=fileDirectory+"DM_Velocity_Distribution-master/f_v_substructure_normalized.txt"
                )
Gaia_sausage._N_esc = 0.9998978371712317



Gaia_sausage_median = DMVelDistr(
                'Gaia_sausage_median',
                '1807.02519 Necib et al.',
                'galacto-centric',
                (3, 'spherical'),
                np.array([117.7, 35.5, -3.1]),
                np.array([108.2, 61.2, 57.7]),
                errors={'mean':(np.array([+1.8, +1.8, +0.9]), np.array([-2.1, -1.8, -0.9])), 'sigma':(np.array([+1.2, +1.5, +0.7]), np.array([-1.3, -1.5, -0.8]))},
                truncated=True,
                path=fileDirectory+"DM_Velocity_Distribution-master/f_v_substructure_normalized.txt"
                )
Gaia_sausage_median._N_esc = 0.9998950931731664



Gaia_MB_helio = DMVelDistr(
                'Gaia_MB_helio',
                '1807.02519 Necib et al.',
                'galacto-centric',
                (1, 'spherical'),
                vcGaia,
                vcGaia/np.sqrt(2.),
                truncated=False,
                path=fileDirectory+"DM_Velocity_Distribution-master/maxwellboltzmann_helio.txt"
                )
Gaia_MB_helio._N_esc = 1.



Nyx_stream = DMVelDistr(
                'Nyx_stream',
                '1907.07190 Necib et al.',
                'galacto-centric',
                (3, 'spherical'),
                np.array([133.90, 130.17, 53.65]),
                np.array([67.13, 45.80, 65.82]),
                errors={'mean':(np.array([1.79, 2.31, 118.79]), np.array([-1.88, -2.40, -114.96])), 'sigma':(np.array([2.43, 1.57, 2.23]), np.array([-2.29, -1.57, -2.04]))},
                truncated=True,
                path=None
                )
Nyx_stream._N_esc = 0.9999953557616433



Nyx_stream_median = DMVelDistr(
                'Nyx_stream_median',
                '1907.07190 Necib et al.',
                'galacto-centric',
                (3, 'spherical'),
                np.array([156.8, 141.0, -1.4]),
                np.array([46.9, 52.5, 70.9]),
                errors={'mean':(np.array([2.1, 2.5, 3.1]), np.array([-2.2, -2.6, -3.])), 'sigma':(np.array([1.7, 1.8, 2.4]), np.array([-1.6, -1.8, -2.2]))},
                truncated=True,
                path=None
                )
Nyx_stream_median._N_esc = 1.0000006156776917



S1_stream = DMVelDistr(
                'S1_stream',
                '1909.04684 O\'Hare et al.',
                'galacto-centric',
                (3, 'cylindrical'),
                np.array([-34.2, -306.3, -64.4]),
                np.array([81.9, 46.3, 62.9]),
                errors={'mean':(np.array([27.92, 21.34, 18.34]), -np.array([27.92, 21.34, 18.34])), 'sigma':(np.array([22.76, 32.37, 23.35]), -np.array([22.76, 32.37, 23.35]))},
                truncated=True,
                path=None
                )
S1_stream._N_esc = 0.9999512628743833



S2a_stream = DMVelDistr(
                'S2a_stream',
                '1909.04684 O\'Hare et al.',
                'galacto-centric',
                (3, 'cylindrical'),
                np.array([5.8, 163.6, -250.4]),
                np.array([45.9, 13.8, 26.8]),
                errors={'mean':(np.array([18.34, 18.52, 19.84]), -np.array([18.34, 18.52, 19.84])), 'sigma':(np.array([17.13, 12.86, 15.66]), -np.array([17.13, 12.86, 15.66]))},
                truncated=True,
                path=None
                )
S2a_stream._N_esc = 1.0053865102478021



S2b_stream = DMVelDistr(
                'S2b_stream',
                '1909.04684 O\'Hare et al.',
                'galacto-centric',
                (3, 'cylindrical'),
                np.array([-50.6, 138.5, 183.1]),
                np.array([90.8, 25.0, 43.8]),
                errors={'mean':(np.array([16.33, 15.21, 20.16]), -np.array([16.33, 15.21, 20.16])), 'sigma':(np.array([21.77, 12.05, 13.66]), -np.array([21.77, 12.05, 13.66]))},
                truncated=True,
                path=None
                )
S2b_stream._N_esc = 1.000176408295106



Toy1_stream = DMVelDistr(
                'Toy1_stream',
                'Manuel\'s imagination',
                'galacto-centric',
                (3, 'cylindrical'),
                v_sun - 330.*eps0,
                25.*np.ones(3),
                truncated=True,
                path=None
                )
Toy1_stream._N_esc = 0.906619684249287



Toy2_stream = DMVelDistr(
                'Toy2_stream',
                'Manuel\'s imagination',
                'galacto-centric',
                (3, 'cylindrical'),
                v_sun + 330.*eps0,
                25.*np.ones(3),
                truncated=True,
                path=None
                )
Toy2_stream._N_esc = 0.9402186375352796



Toy3_stream = DMVelDistr(
                'Toy3_stream',
                'Manuel\'s imagination',
                'galacto-centric',
                (3, 'cylindrical'),
                v_sun + 330.*eps1,
                25.*np.ones(3),
                truncated=True,
                path=None
                )
Toy3_stream._N_esc = 0.9371364767188394



Toy4_stream = DMVelDistr(
                'Toy4_stream',
                'Manuel\'s imagination',
                'galacto-centric',
                (3, 'cylindrical'),
                v_sun - 230.*eps0,
                25.*np.ones(3),
                truncated=True,
                path=None
                )
Toy4_stream._N_esc = 1.0444038405232918



Toy5_stream = DMVelDistr(
                'Toy5_stream',
                'Manuel\'s imagination',
                'galacto-centric',
                (3, 'cylindrical'),
                v_sun - 430.*eps0,
                25.*np.ones(3),
                truncated=True,
                path=None
                )
Toy5_stream._N_esc = 1.0164800109743395



Toy6_stream = DMVelDistr(
                'Toy6_stream',
                'Manuel\'s imagination',
                'galacto-centric',
                (3, 'cylindrical'),
                v_sun - 330.*eps0,
                50.*np.ones(3),
                truncated=True,
                path=None
                )
Toy6_stream._N_esc = 0.995679795338914



Toy7_stream = DMVelDistr(
                'Toy7_stream',
                'Manuel\'s imagination',
                'galacto-centric',
                (3, 'cylindrical'),
                v_sun - 330.*eps0,
                75.*np.ones(3),
                truncated=True,
                path=None
                )
Toy7_stream._N_esc = 0.9633167521402601



Toy8_stream = DMVelDistr(
                'Toy8_stream',
                'Manuel\'s imagination',
                'galacto-centric',
                (3, 'cylindrical'),
                v_sun - 330.*eps0,
                100.*np.ones(3),
                truncated=True,
                path=None
                )
Toy8_stream._N_esc = 0.893370896482584



Sausage_plus = DMVelDistr(
                'Sausage_plus',
                '1807.02519 Necib et al.',
                'galacto-centric',
                (3, 'cylindrical'),
                np.array([117.7, 35.5, -3.1]),
                np.array([108.2, 61.2, 57.7]),
                truncated=True,
                )
Sausage_plus._N_esc = 1.
Sausage_plus._N_esc *= Sausage_plus.integral()



Sausage_minus = DMVelDistr(
                'Sausage_minus',
                '1807.02519 Necib et al.',
                'galacto-centric',
                (3, 'cylindrical'),
                np.array([-117.7, 35.5, -3.1]),
                np.array([108.2, 61.2, 57.7]),
                truncated=True,
                )
Sausage_minus._N_esc = 1.
Sausage_minus._N_esc *= Sausage_minus.integral()



# The dictionary of all the DM components under consideration
dark_matter_components = {
                            'SHM_default':SHM_default,
                            'SHM_infinity':SHM_infinity,
                            'Gaia_halo':Gaia_halo,
                            'Gaia_halo_median':Gaia_halo_median,
                            'Gaia_sausage':Gaia_sausage,
                            'Gaia_sausage_median':Gaia_sausage_median,
                            'Gaia_MB_helio':Gaia_MB_helio,
                            'Nyx_stream':Nyx_stream,
                            'Nyx_stream_median':Nyx_stream_median,
                            'S1_stream':S1_stream,
                            'S2a_stream':S2a_stream,
                            'S2b_stream':S2b_stream,
                            'Toy1_stream':Toy1_stream,
                            'Toy2_stream':Toy2_stream,
                            'Toy3_stream':Toy3_stream,
                            'Toy4_stream':Toy4_stream,
                            'Toy5_stream':Toy5_stream,
                            'Toy6_stream':Toy6_stream,
                            'Toy7_stream':Toy7_stream,
                            'Toy8_stream':Toy8_stream,
                            'Sausage_plus':Sausage_plus,
                            'Sausage_minus':Sausage_minus
                        }

#---------------------------------------------------------------------------------------------------

def fn_time_gen(dm_comp, method, v, vsun=v_sun, t1=tvernal, timeArr=time_arr, **kwargs):
    """
    Generator for the value of a function with vboost evaluated over a time [days] array, for a fixed velocity [km/s].
    
    Parameters
    ----------
    dm_comp : the DM component under consideration
    method : the string name of the method we want to generate
    v : the magnitude of the velocity [km/s]
    vsun : the DM velocity w.r.t. the Sun's frame of reference [km/s] (default: v_sun)
    t1 : an arbitrary initial time [days] (default: tvernal)
    timeArr : a time [days] array (default: time_arr)
    kwargs : other keyword arguments of the function 'method' (v.g. 'taken_from' and 'arrays')
    """
    
    if not (dm_comp in dark_matter_components.keys()): raise NameError("The component \'"+dm_comp+"\' you called is not yet instantiated! Please, either limit yourself to component names listed in "+str(dark_matter_components.keys())+" or, if you really want this component, define it in astrophysics.py")
    
    CompInst = dark_matter_components[dm_comp]
    
    if not (method in dir(CompInst)): raise NameError("The method \'"+method+"\' you called is not does not exist for \'"+dm_comp+"\'. Please limit yourself to the methods in "+str(dir(CompInst))+".")
    
    comp_method = getattr(CompInst, method)
    
    v_lab_t = v_earth(vsun=vsun, t1=t1)[0](timeArr) # the time-dependent vboost in our lab
    
    for vboost_time in v_lab_t:
        yield comp_method(v, vboost=vboost_time, **kwargs)



def fn_vel_gen(dm_comp, method, vArr=v_arr, vboost=v_sun, **kwargs):
    """
    Generator for the value of a function over a velocity [km/s] array, for fixed vboost.
    
    Parameters
    ----------
    dm_comp : the DM component under consideration
    method : the string name of the method we want to generate
    vArr : a velocity [km/s] array (default: v_arr)
    vboost : the boost velocity of the frame under consideration [km/s] (default: v_sun)
    kwargs : other keyword arguments of the function 'method' (v.g. 'taken_from' and 'arrays')
    """
    
    if not (dm_comp in dark_matter_components.keys()): raise NameError("The component \'"+dm_comp+"\' you called is not yet instantiated! Please, either limit yourself to component names listed in "+str(dark_matter_components.keys())+" or, if you really want this component, define it in astrophysics.py")
    
    CompInst = dark_matter_components[dm_comp]
    
    if not (method in dir(CompInst)): raise NameError("The method \'"+method+"\' you called is not does not exist for \'"+dm_comp+"\'. Please limit yourself to the methods in "+str(dir(CompInst))+".")
    
    comp_method = getattr(CompInst, method)
    
    for v in vArr:
        yield comp_method(v, vboost=vboost, **kwargs)



def load_table(dm_comp, method, variables='v', vboost=v_sun, vsun=v_sun, t1=tvernal):
    """
    Loads the values of a function method of a DM component over either velocity or time-velocity grids.
    
    Parameters
    ----------
    dm_comp : the DM component under consideration
    method : the string name of the method we want to generate
    variables : whether 'v' (the average velocity), or 'vt'/'tv' (velocity and time) (default: 'v')
    vboost : the boost velocity of the frame under consideration [km/s] (default: v_sun)
    vsun : the DM velocity w.r.t. the Sun's frame of reference [km/s] (default: v_sun)
    t1 : an arbitrary initial time [days] (default: tvernal)
    """
    
    if variables == 'v':
        if vboost.all() == v_sun.all():
            vboost_name = 'vboost_default'
        else:
            vboost_name = 'vboost_'+str(vboost[0])+"-"+str(vboost[1])+"-"+str(vboost[2])
        
    elif (variables == 'vt' or variables == 'tv'):
        if (vsun.all() == v_sun.all() and t1 == tvernal):
            vboost_name = 'vboost_default'
        else:
            vboost_name = 'vsun_'+str(vsun[0])+"-"+str(vsun[1])+"-"+str(vsun[2])+"_t1_"+str(t1)
    
    else:
        raise ValueError("Value of \'variables\' is not understood. It must be either \'v\' or \'vt\'.")
    
    
    varis = variables if variables!='tv' else 'vt'
    
    if dm_comp in ['SHM_old_March', 'SHM_old_June', 'SHM_old_December']:
        data = fileDirectory+"tables/SHM_old/"+dm_comp+"_"+method+"_Essigs_v.csv"
        
    else:
        
        data = fileDirectory+"tables/"+dm_comp+"/"+dm_comp+"_"+method+"_"+vboost_name+"_"+varis+".csv"
    
    return np.loadtxt(open(data, "r"), delimiter=",")



# The interpolating functions for the different DM components, and different moments of the velocity distribution.
def int_fn_of_avg_v(dm_comp, method, vArr=v_arr, vboost=v_sun, from_scratch=False, **kwargs):
    """
    Interpolation of a method function for a given DM component and fixed vboost, coming from a 1D average velocity [km/s] table. The table is either constructed from scratch, or read from a file.
    
    Parameters
    ----------
    dm_comp : the DM component under consideration
    method : the string name of the method we want to generate
    vArr : a velocity [km/s] array (default: v_arr)
    vboost : the boost velocity of the frame under consideration [km/s] (default: v_sun)
    from_scratch : whether we first have to produce from scratch the table to be interpolated (default: False)
    kwargs : other keyword arguments of the function 'method' (v.g. 'taken_from' and 'arrays')
    """
    
    if from_scratch:
        
        return interp1d(vArr, np.array(list( fn_vel_gen(dm_comp, method, vArr=vArr, vboost=vboost, **kwargs) )), fill_value='extrapolate')
    
    else:
        table = load_table(dm_comp, method, variables='v', vboost=vboost)
        
        if len(table) != len(vArr): raise ValueError("The length of \'vArr\', the velocity array you passed for the interpolation function does not match the length of the table being read. Either compute the array from scratch (set \'from_scratch\'=True) or ensure that the array you passed for \'vArr\' is the same as the one used to produce the table!")
        
        return interp1d(vArr, table, fill_value='extrapolate')
    



def int_fn_vt(dm_comp, method, timeArr=time_arr, vArr=v_arr, vsun=v_sun, t1=tvernal, from_scratch=False, what_inter='interp2d'):
    """
    Interpolation of a function method for a DM component, coming from a 2D time [days]-velocity [km/s] table, for fixed vsun and t1. The table is either constructed from scratch, or read from a file.
    
    Parameters
    ----------
    dm_comp : the DM component under consideration
    method : the string name of the method we want to generate
    vArr : a velocity [km/s] array (default: v_arr)
    timeArr : a time [days] array (default: time_arr)
    vsun : the DM velocity w.r.t. the Sun's frame of reference [km/s] (default: v_sun)
    t1 : an arbitrary initial time [days] (default: tvernal)
    from_scratch : whether we first have to produce from scratch the table to be interpolated (default: False)
    what_inter : what interpolation method we will use: 'interp2d' or 'RBS' (RectBivariateSpline) (default: 'interp2d')
    """
    
    if from_scratch: raise ValueError("!!!!!!!!!!!!!!!!!!\nMODULE UNDER CONSTRUCTION!\n!!!!!!!!!!!!!!!!!!\nAlso, it would take you a long time to build from scratch the table to be interpolated. I recommend you passing a table instead.")
        
    else:
        
        if what_inter == 'interp2d':
            
            table = load_table(dm_comp, method, variables='vt', vsun=vsun, t1=t1)
            
            if table.shape != (timeArr.shape[0], vArr.shape[0]): raise ValueError("\'timeArr\' and \'vArr\', the time and velocity arrays you passed for the interpolation function, do not match the shape of the table being read. Please make sure that the arrays are the same as the ones with which you produced the table!")
            
            return interp2d(x=vArr, y=timeArr, z=table) # call function with fn(x, y) = fn(speed, day)
        
        else:
            
            table = load_table(dm_comp, method, variables='vt', vsun=vsun, t1=t1)
            
            if table.shape != (timeArr.shape[0], vArr.shape[0]): raise ValueError("The time and velocity arrays you passed for the interpolation function do not match the shape of the table being read. Please make sure that the arrays are the same as the ones with which you produced the table!")
        
            return RectBivariateSpline(x=timeArr, y=vArr, z=table)



def init_fns(components, methods=['fv', 'gvmin'], vArr=v_arr, timeArr=time_arr, vboost=v_sun, vsun=v_sun, t1=tvernal, from_scratch=False, **kwargs):
    """
    Returns a dictionary of functions, where the keys are the tuples ('DM Component', 'method/function', 'variables (v, or vt)').
    
    Parameters
    ----------
    components : a list of DM components under consideration
    methods : a list of functions under consideration (default: ['fv', 'gvmin'])
    vArr : a velocity [km/s] array (default: v_arr)
    vboost : the boost velocity of the frame under consideration [km/s] (default: v_sun)
    timeArr : a time [days] array (default: time_arr)
    vsun : the DM velocity w.r.t. the Sun's frame of reference [km/s] (default: v_sun)
    t1 : an arbitrary initial time [days] (default: tvernal)
    from_scratch : whether we first have to produce from scratch the table to be interpolated (default: False)
    kwargs : other keyword arguments of the function 'method' (v.g. 'taken_from' and 'arrays')
    """
    
    fn_dict = {}
    for comp in components:
        for fn in methods:
            fn_dict[(comp, fn, 'v')] = int_fn_of_avg_v(comp, fn, vArr=vArr, vboost=vboost, from_scratch=from_scratch, **kwargs)
            fn_dict[(comp, fn, 'vt')] = int_fn_vt(comp, fn, timeArr=timeArr, vArr=vArr, vsun=vsun, t1=t1, from_scratch=from_scratch)
    
    return fn_dict



def dm_recipe(method, ingredients={'SHM_default':1.}, variables='v', **kwargs):
    """
    Returns a tuple of functions of velocity v [km/s]:
    
    recipe_dict : a dictionary whose keys are the ingredients of the dark matter components and its values are is the function 'func' evaluated on v for the component, weighed by the fraction of the total dark matter said component.
    
    total: the total value of the function 'func' as a function of v [km/s]
    
    Parameters
    ----------
    method : the string name of the method we want to generate
    ingredients : the dictionary of DM components and their fractions (default: {'SHM_default':1.})
    variables : whether 'v' (the average velocity), or 'vt' (velocity and time) (default: 'v')
    kwargs : other keyword arguments of the function 'method'. If 'variables'=='v', then it is 'vArr', 'vboost', 'from_scratch', 'taken_from', and 'arrays'. If 'variables'=='vt' then it is 'timeArr', 'vArr', 'vsun', 't1', and 'from_scratch'.
    """
    
    if variables == 'v':
        
        def each_func(v):
            for dm_comp, frac in ingredients.items():
                yield frac*int_fn_of_avg_v(dm_comp, method, **kwargs)(v)
    
        recipe_dict = lambda v: dict(zip( ingredients.keys(), each_func(v) ))
        total = lambda v: sum([el for el in recipe_dict(v).values()])
    
    
    elif (variables == 'vt' or variables == 'tv'):
        
        def each_func(v, t):
            for dm_comp, frac in ingredients.items():
                yield frac*int_fn_vt(dm_comp, method, **kwargs)(v, t)
        
        recipe_dict = lambda v, t: dict(zip( ingredients.keys(), each_func(v, t) ))
        total = lambda v, t: sum([el for el in recipe_dict(v, t).values()])
    
    
    else:
        raise ValueError("Value of \'variables\' is not understood. It must be either \'v\' or \'vt\'.")
    
    
    return recipe_dict, total
