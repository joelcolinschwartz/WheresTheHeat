# -*- coding: utf-8 -*-
"""


**Put your cleaned-up module docstring above this line.**




New in # 6

Created a subclass called fitter. It'll take results from parcel class, 
stitch together the input time array with 1 period of the parcel time array and 
calculate values for just that one period. It depends only on time, tau_rad and 
wadv. It's quicker... and doesn't return any maps and crap. 

- I removed the unfinished countour shit. I can find them in #5 if i need to finish them and add them later.

- You have to define your fitter object in an awkward way. see fitter.__init__()

TO DO
-----

- make sure this works for negative wadv 

        

- include TESTs for most functions that are easy to run. 
        maybe have a TEST file and in case of TEST = TRUE have the function call that file and 
        draw what it does. 
        
        
-  NEED A GET ITEM METHOD SO WE CAN CHANGE VALUES IN AN OBJECT WITHOUT CHANGING THE WHOLE THING. 
Right ow you can change a value but you have to do the conversions that __init__usually makes by hand. 
Like change ours to seconds or things like that.
"""

"""
This module contains 2 classes. 

-parcel allows you to create a planet object and calculate the planetary 
phase curve for an arbitrary number of orbits (default is 3).  

-fitter is  a subclass of parcel. It takes an arbitrary time array and a parcel object as input 
and stitches it to the default 'parcel' time 
array. It then outputs a planetary phase curve for only the time values in your time array. 
It can be used for fitting for parameters tau_rad and wadv

"""

import numpy as np
import healpy as hp
#import matplotlib.pyplot as plt
from PyAstronomy import pyasl
from scipy import integrate

pi = np.pi

with np.load('planck_integrals_1to10000K_01to30um.npz') as data:
    _preplancked_temperatures = data['temperatures']
    _preplancked_wavelengths = data['wavelengths']
    _preplancked_integrals = data['integrals']

def _make_value_diff_index_array(values):
    """Assumes integers >= 0 only."""
    val_i = values.astype(int)
    # Set max index --> max value
    dvi_vbased = np.zeros(val_i[-1]+1)
    dvi_vbased[val_i] = val_i - np.arange(len(values))
    return dvi_vbased.astype(int)

_preplancked_waveN = len(_preplancked_wavelengths)
# For general method that converts values to indices
_preplancked_temperDvi_vbased = _make_value_diff_index_array(_preplancked_temperatures)


class parcel(object):

    """This class allows you to create a planet object and assign it appropriate orbital 
    and planetary parameters.It uses class functions to calculate the planetary phase curve 
    (emmitted flux) for an arbitrary number of orbits (default is 3). Use this class if you 
    want to make figures. Use the fitter class for fitting.

    Note
    ----
        Need to create a _getitem_ method. At the moment, i can create an obeject and give it properties, but 
        i can't change one of the properties without defining it all over again. 
        
    Object Attributes --  see __init__ documentation 
    ------------------------------------------------
    (instance variables unique to each instance???)

    Params that you might want to change 
    ------------------------------------
        pmax
                int; Number of orbital periods we will integrate for. Default  is 3.
                Might need more for large values of the radiative time scale because you want the DE
                to have time to reach a stable state.  
        steps
                int; number of steps PER 24 hours. Default is 300. This gives nice smooth
                curves for making figures. For fitting, i find that 100 works fine but you can exeriment 
                with that. 
            
        NSIDE
                power of 2; healpix parameter that determines 
                number of pixels that will subdivide the planet surface ;
                NPIX = 12* NSIDE**2.
                (ex: 192 pixels --> NPIX = 192, NSIDE = 4; 798 pixels --> NPIX = 798, NSIDE = 8)
                Default is 8 but 4 is good enough and you should use 4 for fitting or it takes too long.
                
        wavelength
                Values should be entered in micrometers. 
                Used for inc/ emmitted flux calculations at this particular wavelength.
                Default is 8.
                
    """

    astro_unit = (1.49597870700)*(10**11)  # in m
    stef_boltz = (5.670367)*(10**(-8))  # Stefan-Boltzmann constant
    boltz = (1.38064852)*(10**(-23))  # Boltzmann constant
    sec_per_day = 86400
    sec_per_hour = 3600
    
    radius_sun = (6.957)*(10**8)  # in m
    radius_jupiter = (6.9911)*(10**7)  # in m
    mass_sun = (1.98855)*(10**30)  # in kg
    mass_jupiter = (1.8982)*(10**27)  # in kg
    grav_const = (6.67408)*(10**(-11))  # Gravitational constant
    
    planck = (6.62607004)*(10**(-34))
    speed_light = (2.99792458)*(10**8)  # in m/s
    
    _accept_motions = ('calcR','calcA','perR','perA','freqR','freqA')
    
    
    def _setup_scaled_quants(self,Rstar,Mstar,Rplanet,smaxis):
        """Blah blah blah."""
        scl_Rstar = Rstar*self.radius_sun  # star radius
        scl_Mstar = Mstar*self.mass_sun # star mass
        
        scl_Rplanet = Rplanet*self.radius_jupiter  # planet mass
        scl_smaxis = smaxis*self.astro_unit  # semimajor axis
        return scl_Rstar,scl_Mstar,scl_Rplanet,scl_smaxis
    
    
    def _calc_efactor(self,eccen):
        # For scaling ang. vel. at periastron (^-1 for period)
        return ((1-eccen)**(-1.5))*((1+eccen)**0.5)
    
    def _calc_T_irradiation(self):
        """Blah blah blah."""
        return self.Teff*((1-self.bondA)**0.25)*((self.Rstar/(self.smaxis*(1-self.eccen)))**0.5)
    
    
    def _modify_arg_peri(self,ap):
        # KEY!>>>: Argument of periastron measured from ascending node at 1st quarter pahse (alpha = 90 deg).
        #     >>>: But in exoplanet literature, arg. peri. = 90 deg means periastron at TRANSIT (alpha = 0 deg).
        #     >>>: (FYI, the arg. peri. quoted in papers are probably for the host stars.)
        #     >>>: So, we add 180 deg to the input argument for consistency. DON'T GET CONFUSED! :-)
        return (ap+180.0) % 360.0
    

    def _calc_orb_period(self):
        return 2.0*pi*(((self.smaxis**3.0)/(self.Mstar*self.grav_const))**0.5)
    
    def _setup_motion(self,motions,orbval,rotval):
        """Blah blah blah."""
        mot_style,mot_qual = motions[:-1],motions[-1]
        w_to_p = lambda w: np.inf if w == 0 else 2.0*pi/abs(w)
        p_to_w = lambda p,r: 0 if p == np.inf else np.sign(r)*(2.0*pi/p)
        
        if mot_style == 'freq':  # Converted from degrees/day to rad/second
            Porb = (2.0*pi/np.radians(orbval))*self.sec_per_day
            if mot_qual == 'A':
                Wrot = np.radians(rotval)/self.sec_per_day
            elif mot_qual == 'R':
                Wrot = rotval*((2.0*pi/Porb)*self._ecc_factor)
            Prot = w_to_p(Wrot)
        
        else:
            if mot_style == 'calc':  # Calculated in seconds
                Porb = self._calc_orb_period()
            elif mot_style == 'per':  # Converted from days to seconds
                Porb = orbval*self.sec_per_day
            
            if mot_qual == 'A':
                Prot = abs(rotval)*self.sec_per_day
            elif mot_qual == 'R':
                Prot = abs(rotval)*(Porb/self._ecc_factor)
            Wrot = p_to_w(Prot,rotval)
        
        adv_freq_peri = Wrot - ((2.0*pi/Porb)*self._ecc_factor)
        return Porb,Prot,Wrot,adv_freq_peri
    
    
    def _setup_radiate_recirc(self,tau_rad,epsilon):
        """Blah blah blah."""
        if epsilon != None:
            if self.eccen == 0:
                recirc_effic = epsilon
                
                if self.adv_freq_peri == 0:
                    if recirc_effic != 0:
                        print('Constructor warning: atmosphere\'s advective freq. is 0, \"recirc_effic\" is not.')
                        print('    Your planet has no winds, but you want to transport heat.')
                        print('    I am setting radiative time to infinity, but your system is not self-consistent.')
                        print('')
                        radiate_time = np.inf
                    else:
                        radiate_time = 0
            
                else:
                    radiate_time = abs(recirc_effic/self.adv_freq_peri)
                    # Check for mismatched wind direction
                    if abs(np.sign(recirc_effic)-np.sign(self.adv_freq_peri)) == 2:
                        print('Constructor warning: atmosphere\'s advective freq. and \"recirc_effic\" have opposite signs.')
                        print('    Your planet\'s winds flow one way, but you want them going the other way.')
                        print('    Radiative time is defined, but your system is not self-consistent.')
                        print('')

            else:
                print('Constructor ignore: you can only set \"recirc_effic\" for circular orbits.')
                recirc_effic = np.nan
                radiate_time = tau_rad*self.sec_per_hour  # Converted from hours to seconds
            
        else:
            radiate_time = tau_rad*self.sec_per_hour  # Converted from hours to seconds
            re = lambda e: self.adv_freq_peri*radiate_time if e == 0 else np.nan
            recirc_effic = re(self.eccen)
        return radiate_time,recirc_effic


    def _initial_time_array(self):
        """Blah blah blah."""
        t_end = self.Porb*self.numOrbs
        N = round(self.numOrbs*self.stepsPerOrbit)
        timeval = np.linspace(0,t_end,num=N+1)
        return timeval
    
    
    def _setup_the_orbit(self):
        """Ha ha ha."""
        # The KeplerEllipse coordinates are: x == "North", y == "East", z == "away from observer".
        # Our orbits are edge-on with inclination = 90 degrees, so orbits in x-z plane.
        # Longitude of ascending node doesn't really matter, so we set Omega = 0 degrees (along +x axis).
        # Argument of periastron measured from ascending node at 1st quarter phase (alpha = 90 deg).
        # >>> SEE KEY NOTE IN _modify_arg_peri METHOD!!!
        return pyasl.KeplerEllipse(self.smaxis,self.Porb,e=self.eccen,Omega=0.0,w=self.arg_peri,i=90.0)
    
    def _calc_orbit_props(self):
        """Stuff and things."""
        radius = self.kep_E.radius(self.timeval)
        orb_pos,tru_anom = self.kep_E.xyzPos(self.timeval,getTA=True)
        
        # Want alpha(transit) = 0 and alpha(periapsis) = 90 + arg_peri.
        # So: alpha = 90 + arg_peri + tru_anom
        alpha = (90.0 + self.arg_peri + np.degrees(np.array(tru_anom))) % 360.0
        # Minus here because alpha = 0 at transit.
        frac_litup = 0.5*(1.0 - np.cos(np.radians(alpha)))
        
        return radius,orb_pos,tru_anom,alpha,frac_litup
    
    def _find_trans_ecl(self):
        """Blah blah blah."""
        # Periastron happens at t = 0 because "tau" in KeplerEllipse defaults to zero.
        # Get info for conjunctions--transit and eclipse--when orbit crosses y-z plane:
        conjunc_times = np.array(self.kep_E.yzCrossingTime())
        conjunc_pos,conjunc_tru_anom = self.kep_E.xyzPos(conjunc_times,getTA=True)
        # Eclipse has +z (transit -z) so argmax is eclipse index (0 or 1).
        i_ecl = np.argmax(conjunc_pos[:,2])
        
        trans_time = conjunc_times[1-i_ecl]
        trans_pos = conjunc_pos[1-i_ecl]
        trans_tru_anom = conjunc_tru_anom[1-i_ecl]
        ecl_time = conjunc_times[i_ecl]
        ecl_pos = conjunc_pos[i_ecl]
        ecl_tru_anom = conjunc_tru_anom[i_ecl]
        
        return trans_time,trans_pos,trans_tru_anom,ecl_time,ecl_pos,ecl_tru_anom
    
    
    def _setup_colatlong(self,NSIDE):
        """Blah blah blah."""
        colat,longs = hp.pix2ang(NSIDE,list(range(hp.nside2npix(NSIDE))))
        pixel_sq_rad = hp.nside2pixarea(NSIDE)
        return colat,longs,pixel_sq_rad,NSIDE
    
    def _calc_longs(self):
        """Blah blah blah."""
        # Planet coordinates: longitude = 0 always points at star.
        # Longitude of gas parcels change throughout orbit. Colatitude stays the same.
        # So: new_longs = orig_longs +- Rotation effect (East/West) - Orbit effect (West)
        new_longs = self.longs[np.newaxis,:] + (self.Wrot*self.timeval[:,np.newaxis]) - self.tru_anom[:,np.newaxis]
        longs_evolve = new_longs % (2.0*pi)
        return longs_evolve
    
    def _calc_vis_illum(self):
        """Blah blah blah."""
        # Sub-stellar point: always longitude = 0 in our coordinates.
        SSP_long = 0
        longs_minus_SSP = self.longs_evolve - SSP_long
        
        # Sub-observer point: get longitude from alpha (orbital phase for observer)
        # SOP_long = pi when alpha = 0, and Westward drift means -alpha.
        SOP_long = (pi - np.radians(self.alpha)) % (2.0*pi)
        longs_minus_SOP = self.longs_evolve - SOP_long[:,np.newaxis]
        
        illumination = 0.5*(np.cos(longs_minus_SSP) + np.absolute(np.cos(longs_minus_SSP)))*np.sin(self.colat)
        visibility = 0.5*(np.cos(longs_minus_SOP) + np.absolute(np.cos(longs_minus_SOP)))*np.sin(self.colat)
        
        return illumination,visibility,SSP_long,SOP_long
    

    def _initial_temperature_array(self):
        """Something something else."""
        Tvals_evolve = np.zeros(self.longs_evolve.shape)
        
        the_low_case = (0.5*(np.cos(self.longs) + np.absolute(np.cos(self.longs)))*np.sin(self.colat))**0.25
        the_high_case = (np.sin(self.colat)/pi)**0.25
        if np.isnan(self.recirc_effic):
            pos_eps = abs(self.adv_freq_peri*self.radiate_time)
        else:
            pos_eps = abs(self.recirc_effic)  # Negative recirc_effic's don't work in "the_scaler".
        the_scaler = (pos_eps**1.652)/(1.828 + pos_eps**1.652)  # Estimated curly epsilon, Schwartz et al. 2017
        Tvals = the_scaler*the_high_case + (1.0-the_scaler)*the_low_case  # E.B. model parameterization
        Tvals[Tvals<0.01] = 0.01
        
        Tvals_evolve[0,:] += Tvals
        return Tvals_evolve

    
    def __init__(self,name='Hot Jupiter',Teff=5778,Rstar=1.0,Mstar=1.0,
                 Rplanet=1.0,smaxis=0.1,eccen=0,arg_peri=0,bondA=0,
                 motions='calcR',orbval=1.0,rotval=1.0,
                 radiate_time=12.0,recirc_effic=None,
                 numOrbs=3,stepsPerRot=360,NSIDE=8):
        
        """The __init__ method allows to set attributes unique to each parcel instance.
        It takes some parameters in the units specified in the docstring. Some are 
        converted to SI units, some are calculated from a few parameters. It has some default
        values that dont have any particular meaning. 
        
        
        Note
        ----
        Some parameters are not used in calculations. They're just there is case i need them for something in the future. 

        Parameters
        ----------
        
        name (str): can give it a name if you want 

        Teff (float): 
            Temperature of star (K)
        
        Rstar (float): 
            Radius of star (in units of solar radii)
        
        Mstar (float): 
            mass of the star in solar masses
        
        Rplanet (float): 
            Radius of planet in Jupiter Radii
        
        a (float): 
            Semimajor axis in AU 
        
        e (float, 0 to 1):
            eccentricity
        
        argp (float):
            Argument at periastron in degrees - angle betwen periastron and transit in degrees 

        A : Bond albedo (set to 0)
            Model doesn not handle reflected light right now. Setting albedo to a different value 
            will have no effect than to reduce incoming flux by a certain fraction.
        
        Porb (float): orbital period in days 
            will be calculated from Kepler's laws if Porb param set to -1    
            
            
        P : 
            Rotational period of the gas around the planet; calculated by self.Prot(). 
            
  
        wadv : PARAM WE WOULD FIT FOR
            
            multiple of wmax (ex: 2 or 0.5)
            wrot = (2*Pi/P) is chosen to match wmax (orbital angular velocity at periastron );
            wadv is expressed as a multiple of wmax, with ( - ) meaning a rotation in the oposite direction.

            PROBLEM/ REMARK: 
                in the circular case: if wadv = 1 it means the gas isnt moving wrt the substellar point. 
                Every parcel of gas has the same temperature it started with always 
                (might heat up a bit and stay there)
                
                if wadv = 2 it means that it takes 1 rotaions of the planet for the gas to leave the Substellar point 
                and come back (go through all its temperatures)
                
                in the eccentric case: this is more complicated. If wadv = 1 it can be interpreted as 
                the substellar point and the gas being stationary wrt each other at periastron.
 
        T0 : not used in calculations
            Initial temperature of gas at substellar point at periastron. 
            
            Teff*(1-A)**(0.25)*(self.Rstar/(self.a*(1-self.e)))**(0.5) 

        tau_rad (and epsilon): PARAM WE WOULD FIT FOR
            
            epsilon = tau_rad * wadv

            For eccentric orbit, value for tau_rad should be entered in hours and epsilon
            should be left blank. 

            For a circular orbit, epsilon (efficiency parameter) and wadv should be provided 
            and tau_rad left blank.              

        
        rotationsPerOrbit : np.ceil(max(self.Porb/self.Prot),1)
            
            used for giving the default time lenght for DE
        
        rotationsPerDay : int(self.Prot/self.sec_per_day)
            
            used for giving the default precision for DE
            
        pmax (int)
                Number of orbital periods we will integrate for.
        
        steps (int)
                number of steps PER 24 hours.
                

        NSIDE
                power of 2; healpix parameter that determines 
                number of pixels that will subdivide the planet surface ;
                NPIX = 12* NSIDE**2.
                (ex: 192 pixels --> NSIDE = 4; 798 pixels --> NSIDE = 8)
                
                
        Precalculated quantities that get attached to the object
        --------------------------------------------------------
        
        t -time array (1D)
                
        radius -orbital separation array
        
        ang_vel - orbital angular velocity array
        
        alpha - phase angle array
        
        f - illuminted fraction array
        
        phis, thetas - initial pixel coordinates. starting point for each gas parcel
        on the planet
        
        """
        
        self.name = name
        
        self.Rstar,self.Mstar,self.Rplanet,self.smaxis = self._setup_scaled_quants(Rstar,Mstar,Rplanet,smaxis)
        
        self.eccen = eccen  # eccentricity
        self._ecc_factor = self._calc_efactor(eccen)  # For scaling ang. vel. at periastron (^-1 for period)
        
        self.bondA = bondA  # planet Bond albedo
        self.Teff = Teff  # star effective temp
        self.Tirrad = self._calc_T_irradiation()
        
        ### We add 180 deg to the input "arg_peri", see this method. DON'T GET CONFUSED! :-)
        self.arg_peri = self._modify_arg_peri(arg_peri)
        
        if motions not in self._accept_motions:
            print('Constructor error: \"motions\" should be one of these strings:')
            print(self._accept_motions)
            return
        self.Porb,self.Prot,self.Wrot,self.adv_freq_peri = self._setup_motion(motions,orbval,rotval)
        
        self.radiate_time,self.recirc_effic = self._setup_radiate_recirc(radiate_time,recirc_effic)
        
        ### Time
        self.rotationsPerOrbit = self.Porb/self.Prot  # If Prot = 0 it's your own fault. :-)
        # When Porb < Prot, stepsPerRot will be steps per orbit.
        self.stepsPerOrbit = round(stepsPerRot*max(self.rotationsPerOrbit,1.0))
        self.numOrbs = numOrbs  # Total orbits for time array
        self.timeval = self._initial_time_array()
        
        ### Orbital stuff
        self.kep_E = self._setup_the_orbit()
        
        (self.radius,self.orb_pos,
         self.tru_anom,self.alpha,self.frac_litup) = self._calc_orbit_props()
        (self.trans_time,self.trans_pos,self.trans_tru_anom,
         self.ecl_time,self.ecl_pos,self.ecl_tru_anom) = self._find_trans_ecl()
        
        ### Atmosphere coordinates
        self.colat,self.longs,self.pixel_sq_rad,self.NSIDE = self._setup_colatlong(NSIDE)
        self.longs_evolve = self._calc_longs()
        
        (self.illumination,self.visibility,
         self.SSP_long,self.SOP_long) = self._calc_vis_illum()
        
        ### Temperatures
        self.Tvals_evolve = self._initial_temperature_array()
        return
        

    def Info_Print(self):
        """Blah blah blah."""
        # Name
        print('Below are some parameters you are using to model {}.'.format(self.name))
        print('')
        
        # Rstar, Mstar, Teff
        form_cols = '{:^16} {:^16} {:^18}'
        print(form_cols.format('R_star (solar)','M_star (solar)','T_effective (K)'))
        form_cols = '{:^16.2f} {:^16.2f} {:^18.1f}'
        print(form_cols.format(self.Rstar/self.radius_sun,self.Mstar/self.mass_sun,self.Teff))
        print('')
        
        # Rplanet, Bond, smaxis, T0
        form_cols = '{:^20} {:^14} {:^16} {:^20}'
        print(form_cols.format('R_planet (Jupiter)','Bond albedo','Semimajor (AU)','T_irradiation (K)'))
        form_cols = '{:^20.2f} {:^14.2f} {:^16.3f} {:^20.1f}'
        print(form_cols.format(self.Rplanet/self.radius_jupiter,self.bondA,self.smaxis/self.astro_unit,self.Tirrad))
        print('')
        
        # Porb, Prot, eccen, argp
        form_cols = '{:^14} {:^14} {:^14} {:^24}'
        print(form_cols.format('P_orb (days)','P_rot (days)','Eccentricity','Arg. periastron (deg)'))
        form_cols = '{:^14.2f} {:^14.2f} {:^14.3f} {:^24.1f}'
        print(form_cols.format(self.Porb/self.sec_per_day,self.Prot/self.sec_per_day,self.eccen,self.arg_peri))
        print('')
        
        # adv_freq_peri, radiate_time, recirc_effic
        form_cols = '{:^26} {:^22} {:^10}'
        print(form_cols.format('Advective freq. (rad/hr)','Radiative time (hrs)','Epsilon'))
        form_cols = '{:^26.3f} {:^22.3f} {:^10.3f}'
        print(form_cols.format(self.adv_freq_peri*self.sec_per_hour,self.radiate_time/self.sec_per_hour,self.recirc_effic))
        
        return
    
    
    ### Differential Equation
    
    def _initial_temperatures(self):
        """Something something else."""
        the_low_case = (0.5*(np.cos(self.longs) + np.absolute(np.cos(self.longs)))*np.sin(self.colat))**0.25
        the_high_case = (np.sin(self.colat)/pi)**0.25
        if np.isnan(self.recirc_effic):
            pos_eps = abs(self.adv_freq_peri*self.radiate_time)
        else:
            pos_eps = abs(self.recirc_effic)  # Negative recirc_effic's don't work in "the_scaler".
        the_scaler = (pos_eps**1.652)/(1.828 + pos_eps**1.652)  # Estimated curly epsilon, Schwartz et al. 2017
        Tvals = the_scaler*the_high_case + (1.0-the_scaler)*the_low_case  # E.B. model parameterization
        Tvals[Tvals<0.01] = 0.01
        return Tvals
    
    
    def _diff_eq_tempvals(self,start_Tvals):
        """Something something else."""
        Tvals_evolve = np.zeros(self.longs_evolve.shape)
        Tvals_evolve[0,:] += start_Tvals
        
        if self.eccen == 0:
            if (abs(self.recirc_effic) <= 10**(-4)):
                Tvals_evolve = ((1.0-self.bondA)*self.illumination)**(0.25)
            else:
                # Here advective frequency is constant- sign spcifies direction atmosphere rotates.
                sn = lambda w: -1.0 if w < 0 else 1.0
                delta_longs = (self.longs_evolve[1:,:] - self.longs_evolve[:-1,:]) % (sn(self.adv_freq_peri)*2.0*pi)
                
                for i in range(1,len(self.timeval)):
                    # Stellar flux is constant for circular orbits, F(t)/Fmax = 1.
                    delta_Tvals = (1.0/self.recirc_effic)*(self.illumination[i-1,:] - (Tvals_evolve[i-1,:]**4))*delta_longs[i-1,:]
                    Tvals_evolve[i,:] = Tvals_evolve[i-1,:] + delta_Tvals  # Step-by-step T update
    
        else:
            # Normalized stellar flux
            scaled_illum = self.illumination*((self.smaxis*(1-self.eccen)/self.radius[:,np.newaxis])**2)
            
            # Eccentric DE uses t_tilda = t/radiate_time
            if self.radiate_time <= 10**(-4):
                Tvals_evolve = ((1.0-self.bondA)*scaled_illum)**(0.25)
            else:
                delta_radtime = (self.timeval[1:] - self.timeval[:-1])/self.radiate_time
                
                for i in range(1,len(self.timeval)):
                    delta_Tvals = (scaled_illum[i-1,:] - (Tvals_evolve[i-1,:]**4))*delta_radtime[i-1]
                    Tvals_evolve[i,:] = Tvals_evolve[i-1,:] + delta_Tvals  # Step-by-step T update
        
        return Tvals_evolve
    
    def Evolve_AtmoTemps(self):
        """Something something else."""
        if True:
            start_Tvals = self._initial_temperatures()
        else:
            pass
    
        self.Tvals_evolve = self._diff_eq_tempvals(start_Tvals)
        
        print('Evolving complete')
        return
    

    ### Blackbody methods
    
    def _waveband_to_lowup(self,wave_microns,band_microns):
        """Blah blah blah."""
        return wave_microns-(0.5*band_microns),wave_microns+(0.5*band_microns)
    
    def _um_to_m(self,microns):
        """Blah blah blah."""
        return microns*(10**(-6))

    
    def _blackbody_bolo(self,temperature):
        """Units of W/m^2"""
        return self.stef_boltz*(temperature**4)


    def _plancks_law(self,wavelength,temperature):
        """Blah blah blah."""
        xpon = self.planck*self.speed_light/(wavelength*self.boltz*temperature)
        # Leading pi from integral over solid angle, so wave/bolo units match.
        return pi*(2.0*self.planck*(self.speed_light**2)/(wavelength**5))*(1.0/(np.exp(xpon) - 1.0))
    
    def _integral_plancks(self,lower_microns,upper_microns,temperature):
        """Blah blah blah."""
        lower_wave,upper_wave = self._um_to_m(lower_microns),self._um_to_m(upper_microns)
        return integrate.quad(self._plancks_law,lower_wave,upper_wave,args=(temperature))

    def _blackbody_wavelength(self,lower_microns,upper_microns,temperature):
        """Units of W/m^2"""
        vec_integral_plancks = np.vectorize(self._integral_plancks)
        bb_values,bb_errors = vec_integral_plancks(lower_microns,upper_microns,temperature)
        return bb_values,bb_errors
    
    
    def _get_wave_indices(self,lower_microns,upper_microns):
        """Blah blah blah."""
        lower_i = np.argmin(np.absolute(_preplancked_wavelengths - lower_microns))
        upper_i = np.argmin(np.absolute(_preplancked_wavelengths - upper_microns))
        if lower_i == upper_i:
            if upper_i == (_preplancked_waveN - 1):
                lower_i -= 1
            else:
                upper_i += 1
        return lower_i,upper_i
    
    def _change_temper_vals_to_inds(self,temper_v):
        """Cool, looks like this works well!"""
        tv_asi = temper_v.astype(int)
        return tv_asi - _preplancked_temperDvi_vbased[tv_asi]
    
    def _get_temper_indices(self,temperature):
        """Blah blah blah."""
        high_t = _preplancked_temperatures[-1]
        low_t = _preplancked_temperatures[0]
        
        if isinstance(temperature,np.ndarray):
            temper_v = np.around(temperature)
            high_check = (temper_v > high_t)
            temper_v[high_check] = high_t
            low_check = (temper_v < low_t)
            temper_v[low_check] = low_t
        else:
            check = lambda t: high_t if t > high_t else (low_t if t < low_t else t)
            temper_v = np.around(check(temperature))
        
        temper_i = self._change_temper_vals_to_inds(temper_v)
        return temper_i

    def _blackbody_preplancked(self,lower_microns,upper_microns,temperature):
        """Units of W/m^2"""
        lower_i,upper_i = self._get_wave_indices(lower_microns,upper_microns)
        temper_i = self._get_temper_indices(temperature)

        chosen_integrals = np.sum(_preplancked_integrals[:,lower_i:upper_i],axis=1)
        return chosen_integrals[temper_i]
    

    def Observed_Flux(self,wave_band=False,a_microns=6.5,b_microns=9.5,
                      kind='obs',run_integrals=False,bolo=False):
        """Blah blah blah."""
        if wave_band:
            lower_microns,upper_microns = self._waveband_to_lowup(a_microns,b_microns)
        else:
            lower_microns,upper_microns = a_microns,b_microns
        
        if kind == 'obs':
            star_vis = pi
            planet_vis = self.visibility
        elif kind == 'hemi':
            star_vis = 2.0*pi
            planet_vis = (self.visibility > 0)
        elif kind == 'sphere':
            star_vis = 4.0*pi
            planet_vis = 1.0

        planet_Treal = self.Tvals_evolve*self.Tirrad
        if bolo:
            star_bb = self._blackbody_bolo(self.Teff)
            planet_bb = self._blackbody_bolo(planet_Treal)
        elif run_integrals:
            star_bb,_ignore = self._blackbody_wavelength(lower_microns,upper_microns,self.Teff)
            planet_bb,_ignore = self._blackbody_wavelength(lower_microns,upper_microns,planet_Treal)
        else:
            # Warning if Teff or planet_Treal > preplancked T's?
            star_bb = self._blackbody_preplancked(lower_microns,upper_microns,self.Teff)
            planet_bb = self._blackbody_preplancked(lower_microns,upper_microns,planet_Treal)

        star_flux = (star_vis*star_bb)*(self.Rstar**2)
        planet_flux = (np.sum(planet_vis*planet_bb,axis=1)*self.pixel_sq_rad)*(self.Rplanet**2)
        return planet_flux/star_flux


    
    ### ### ### ### ###
    

    def Fstar(self,microns=8.0):

        """Calculates total flux emmitted by the star and wavelength dependent flux.
        
        Used for normalizing planet flux. (i.e. to express flux emmitted by the planet as 
        a fraction of stellar flux. )
        
        Note
        ----
             F = stef_boltz * Teff**4 * Pi * Rstar**2.
             Fwv = BBflux(wv) * Pi * Rstar**2
             
    
        Parameters
        ----------
            wv (float)
                wavelength we are interested in micrometers.

            
        Returns
        -------
            
                F (float)
                    Total flux
                
                Fwv (float)
                    Flux emmitted at the specified wavelength.

        """
        bolo_blackbody = self.stef_boltz*(self.Teff**4)
        wavelength = microns*(10**(-6))  # microns to meters
        xpon = self.planck*self.speed_light/(wavelength*self.boltz*self.Teff)
        ## So bolo and wave units match, leading "pi" is from integral over solid angle.
        wave_blackbody = pi*(2.0*self.planck*(self.speed_light**2)/(wavelength**5))*(1.0/(np.exp(xpon) - 1.0))
        
        return bolo_blackbody*(4.0*pi*(self.Rstar**2)),wave_blackbody*(4.0*pi*(self.Rstar**2))
        
        
    def Finc(self):
        """Total (all wavelengths) flux incident on substellar point as 
        a function of time (position in the orbit).
        
        Used in self.DE for '"incoming energy" value as (Finc/ Finc_max)*Fweight;
        Fweight comes from self.illum
        

        Note
        ----
             stef_boltz*Teff**4*(Rstar/r(t))**2.
        
        Parameters 
        ----------
            None. 
            
        Returns
        -------
            
                Finc (1D array)
                    Theoretical total flux incident on substellar point at each moment on time;
                    lenght pmax * int(Porb/ 24hrs)*steps. 

        """
        ## Only scales sigma*T^4, may need to adjust later.
        return self.stef_boltz*(self.Teff**4)*((self.Rstar/self.radius)**2)


    def DE(self):
        """DE that calculates temperature of each gas parcel as a
        function of time . Relies on self.illum() to
        pass it a time array and coordinates.

        Note
        ----
        Solves the DE by repeatedly adding dT to previous T value.
        Might want to change this to a more sophisticated
        differential equation solver.
        

        Parameters
        ----------
        None
        
        Calls
        -------
        
        self.illum (pmax, steps, NSIDE) to get:
            t
                1D time array of lenght pmax * int(Porb/ 24hrs)*steps;
                in seconds
                (if called by a fitter object it'll take the custom time array)
            
            d
                3D position and temperature array;
                shape = (len(time), NPIX, 3)
                
                3 refers to the 3 columns :
                d[:,:,0] = thetas - latitude -- 0 to Pi
                
                d[:,:,1] = phis - longitude -- 0 to Pi
                (at t = 0, the planet is at periastron and the substellar point
                is located at theta = Pi	/2, phi = 0)
                
                d[:,:,2] = starting temperature array
        
        
        Returns
        -------
        t
            if called by fitter object, will return the backend of the time array
            (i.e. the part you need for fitting)
            
            if called by parcel object will return t unchanged
            
        d
            will return the part of the array that matches t, depending on who's calling the function.
            
            only other change is to replace the starting temperature values
            with values calculated by the DE
            

        """
        if self.eccen == 0:
            if (abs(self.recirc_effic) <= 10**(-4)) or (self.radiate_time <= 10**(-4)):
                self.Tvals_evolve = ((1.0-self.bondA)*self.illumination)**(0.25)
            else:
                # Here advective frequency is constant- sign spcifies direction atmosphere rotates.
                sn = lambda w: -1.0 if w < 0 else 1.0
                delta_longs = (self.longs_evolve[1:,:] - self.longs_evolve[:-1,:]) % (sn(self.adv_freq_peri)*2.0*pi)
                
                for i in range(1,len(self.timeval)):
                    # Stellar flux is constant for circular orbits, F(t)/Fmax = 1.
                    delta_Tvals = (1.0/self.recirc_effic)*(self.illumination[i-1,:] - (self.Tvals_evolve[i-1,:]**4))*delta_longs[i-1,:]
                    self.Tvals_evolve[i,:] = self.Tvals_evolve[i-1,:] + delta_Tvals  # Step-by-step T update
        else:
            # Normalized stellar flux, can clean up becasue A LOT cancels.
#            flux_inc = self.Finc()[:,np.newaxis]
#            flux_max = self.stef_boltz*(self.Teff**4)*((self.Rstar/(self.smaxis*(1.0-self.eccen)))**2)
#            scaled_illum = (flux_inc/flux_max)*self.illumination
            scaled_illum = self.illumination*((self.smaxis*(1-self.eccen)/self.radius[:,np.newaxis])**2)
            
            # Eccentric DE uses t_tilda = t/radiate_time
            if self.radiate_time <= 10**(-4):
                self.Tvals_evolve = ((1.0-self.bondA)*scaled_illum)**(0.25)  # Why divided by stef_boltz before??
            else:
                delta_radtime = (self.timeval[1:] - self.timeval[:-1])/self.radiate_time
                
                for i in range(1,len(self.timeval)):
                    delta_Tvals = (scaled_illum[i-1,:] - (self.Tvals_evolve[i-1,:]**4))*delta_radtime[i-1]
                    self.Tvals_evolve[i,:] = self.Tvals_evolve[i-1,:] + delta_Tvals  # Step-by-step T update
                    
        return
    
    
    def Fleaving(self,microns=8.0):
        """Calculates outgoing planetary flux (Total and wavelength dependant)
        from the temperature values coming from the DE. 

            Note
            ----
            Has an overflow problem sometimes, especially for tau_rad ~ 0. That causes nightside 
            temperatures to be close to 0 and the division in BB flux blows up.
        
    
            Parameters
            ----------
                
            wavelength
                in micrometers; wavelength to calculate the flux at. 
                
            MAP:
                Default is False.
                Use True if you want to draw flux leaving from the planet on a map.
                
                
            Calls
            -------
            
            self.DE (pmax, steps, NSIDE) to get:
                t
                    1D time array 
                
                d 
                    3D position and temperature array;
                    shape = (len(time), NPIX, 3)
                    
                    3 refers to the 3 columns : 
                    d[:,:,0] = thetas - latitude -- 0 to Pi	 
                    
                    d[:,:,1] = phis - longitude -- 0 to Pi	
                    (at t = 0, the planet is at periastron and the substellar point 
                    is located at theta = Pi	/2, phi = 0)
                    
                    d[:,:,2] = surface temperature array
                    
            self.shuffle(d, Fmap_wv, NSIDE) to get 
                
                Fmap_wvpix
                    2D flux array rearraged so the pixels are drawn at the right spots. 
                    see shuffle()
                    
            
            
            Returns
            -------
            If MAP = False:
                t
                unchanged
                
                d 
                    unchanged
                    
                Fmap_wv 
                    2D array[time, NPIX flux values]. outgoing flux map
                    
            If MAP = True:

                t
                    unchanged
                    
                d 
                    unchanged
                    
                Fmap_wv 
                    2D array[time, NPIX flux values]. outgoing flux map 
                    
                Fmap_wvpix 
                    2D array[time, NPIX flux values]. outgoing flux map rearraged 
                    for drawing. see shuffle() 
                    
                Fleavingwv
                    1D array, contains  flux (wavelength dependant) integrated over planet surface
                    at each moment in time. This isnt super useful because you wouldnt 
                    be able to see all the flux coming from the planet.
                    
                Ftotal
                    1D array, contains flux (all wavelengths) integrated over planet surface
                    at each moment in time. Again, not super useful unless you're making figures.
   
        """
        self.DE()  # May want to restructure this.
        
        bolo_blackbody = self.stef_boltz*((self.Tvals_evolve*self.Tirrad)**4)
        wavelength = microns*(10**(-6))  # microns to meters
        xpon = self.planck*self.speed_light/(wavelength*self.boltz*(self.Tvals_evolve*self.Tirrad))
        ## So bolo and wave units match, leading "pi" is from integral over solid angle.
        wave_blackbody = pi*(2.0*self.planck*(self.speed_light**2)/(wavelength**5))*(1.0/(np.exp(xpon) - 1.0))
        
        return bolo_blackbody,wave_blackbody
    
    
    def Fobs(self,microns=8.0,hemi=True,sphere=True):  ###JCS: Added 'HEMISUM' 02/15/17...and now it's gone ;-)
        """ Calculates outgoing planetary flux as seen by an observer (wavelength dependant only).
        

            Note
            ----
            THIS IS THE FUNCTION THAT WILL GIVE YOU THE LIGHT CURVE. 
            
            Remark: i don't totally trust the shuffle function. But it's only used for 
            drawing stuff right now, the outgoing flux is calculated without it.
        
    
            Parameters
            ----------

            wavelength
                in micrometers; wavelength to calculate the flux at. 
                
            PRINT (bool):
                Option to print the results to a text file. Only works if MAP is also True.
                
            MAP (bool):
                Option to return a bunch of stuff i use for making figures.
                
            Calls
            -------
            
            self.Fleaving(wavelength) to get :

                Fmap_wv 
                    2D array[time, NPIX flux values]. outgoing flux map 
                    
                Fmap_wvpix 
                    2D array[time, NPIX flux values]. outgoing flux map rearraged 
                    for drawing. see shuffle() 

                
            self.visibility(d) to get:
                t
                    time array in seconds
                    
                d
                    3D coordinate and temperature array
                    
                weight
                    2D array; [time, position (angle)] ; visibility function


            self.shuffle(d) to get:
                
                weightpix
                    2D array; [time, position(pixel number)] ; rearranged visibility 
                    function for drawing with hp.mollview(). 

   
            Returns
            -------
            If MAP = True:
            t, d, Fmapwvobs, weight, weightpix, Fwv

            t
                unchanged
                
            d 
                unchanged
                
            Fmapwvobs 
                2D array[time, NPIX flux values]; outgoing flux map; 
                rearranged for drawing on a hp.mollview map. 
                *you want this one
                
            weight 
                2D array; [time, position (angle)] ; visibility function
                
            weightpix
                2D array; [time, position(pixel number)] ; rearranged visibility 
                function for drawing with hp.mollview().  
                
            Fwv
                1D array, contains observed flux (wavelength dependant) integrated over planet surface
                at each moment in time.
                *and this one 
                
                if Print = TRUE:
                    also makes a text file with all these arrays. This is actually not useful, 
                    the thing runs fast enough that you dont need to save results. 
                    
                
            
            If Map = False (default)
            
            t, d, Fwv
        """
        bolo_blackbody,wave_blackbody = self.Fleaving(microns)
        
        unit_area = hp.nside2pixarea(self.NSIDE)*(self.Rplanet**2)
        
        # Observer flux (i.e. light curve)
        flux_bolo_obs = np.sum(self.visibility*bolo_blackbody*unit_area,axis=1)
        flux_wave_obs = np.sum(self.visibility*wave_blackbody*unit_area,axis=1)
        
        # Hemispere flux (i.e. without observer visibility)
        hemi_visible = (self.visibility > 0)
        flux_bolo_hemi = np.sum(hemi_visible*bolo_blackbody*unit_area,axis=1)
        flux_wave_hemi = np.sum(hemi_visible*wave_blackbody*unit_area,axis=1)
        
        # Planet flux (i.e. over whole sphere)
        flux_bolo_planet = np.sum(bolo_blackbody*unit_area,axis=1)
        flux_wave_planet = np.sum(wave_blackbody*unit_area,axis=1)
        
        # Probably want to choose returns at some point.
        return flux_bolo_obs,flux_wave_obs,flux_bolo_hemi,flux_wave_hemi,flux_bolo_planet,flux_wave_planet

    
    def Calc_MaxDuskDawn_Temps(self):
        """Something.
            
        Something else.
        
        """
        start_time = int(self.stepsPerOrbit*(self.numOrbs-1))
        
        # For now, each gives you values at every time step.
        # Can do an extra check to get a single value.
        max_args = np.argmax(self.Tvals_evolve[start_time:,:],axis=1)
        max_angles = hp.pix2ang(self.NSIDE,max_args)
        
        Ttilda_max = np.amax(self.Tvals_evolve[start_time:,:],axis=1)
        # DUSK AND DAWN NEED WORK BECAUSE YOU FIXED THE COORDINATE SYSTEM TO THE SSP!!!
        Ttilda_dusk = self.Tvals_evolve[start_time:,hp.ang2pix(self.NSIDE,pi/2.0,pi/2.0)]
        Ttilda_dawn = self.Tvals_evolve[start_time:,hp.ang2pix(self.NSIDE,pi/2.0,-pi/2.0)]
    
        return max_angles,Ttilda_max,Ttilda_dusk,Ttilda_dawn
    
#    def findT (self):
#        """ Finds numeric approximation of Max/ Min temperature on the planet.
#        !!! DOES NOT WORK AS EXPECTED!!! should fix
#
#        Note
#        ----
#
#        ONLY WORKS FOR CIRCULAR ORBITS.
#
#        Used for testing. Supposed to compare to the analytic approximations in
#        the functions phi-max, Tmax, Tdusk, Tdawn, to
#        check that the DE is working well. Or to check that the analytic approx.
#        is working well.
#
#        Parameters
#        ----------
#        None
#
#        Calls
#        -------
#
#        self.DE(), the 0 eccentricity branch.
#
#
#
#        Returns
#        -------
#
#        Tmax (float)
#            Maximum temperature on the planet in T/T0
#
#        Tdawn
#            Dawn temperature on the planet in T/T0
#
#        Tdusk
#            Dusk temperature on the planet in T/T0
#
#        """
#
#        #tmax = self.Prot*pmax
#        #Nmin = int((pmax)*300)
#        #deltat = tmax/Nmin
#        pmaxi = self.pmaxi
#        stepsi = self.stepsi
#
#        t,d = self.DE()
#
#
#        #deltaphi = 2.0*pi/stepsi
#        Tmax = np.max(np.max(d[int(self.stepsi*(self.pmaxi-1))::,:,2],axis =1))
#        #Tmax = np.max(d[int(stepsi*(pmaxi-2))::,:,2])
#
#
#        #for i in range(int(stepsi*(pmaxi-2)), int(stepsi*pmaxi)):
#
#
#
#                #if deltaphi >= np.abs(1.5*pi - (d[i,np.where(np.abs(d[i,:,0]-0.5*pi))< 0.1, np.where(np.abs(d[i,:,1]-1.5*pi))< 0.1]):
#                    #print np.abs(1.5 - phi[i])*pi, 'dawn difference'
#                    #Tdawn = T[i]
#        Tdawn = (d[int(stepsi*(pmaxi-1)),hp.ang2pix(self.NSIDE, pi/2, -pi/2),2])
#
#        Tdusk = (d[int(stepsi*(pmaxi-1)),hp.ang2pix(self.NSIDE, pi/2, pi/2),2])
#
#
#                #if deltaphi >= np.abs(2.5 - (phi[i]-2*(pmaxi-2))):
#                    #print np.abs(2.5 - phi[i])*pi, 'dusk difference'
#
#                    #Tdusk = T[i]
#
#        return Tmax, Tdawn, Tdusk

    
    def _max_temper(self,eps):
        """Something."""
        x_0 = 2.9685
        x_1 = 7.0623
        x_2 = 1.1756
        x_3 = -0.2958
        x_4 = 0.1846
        xpon = -x_2 + (x_3/(1.0 + (x_4*eps)))
        func = x_0/(1.0 + x_1*(eps**xpon))
        return np.cos(np.arctan(func))**(0.25)
    
    def _dusk_temper(self,eps):
        """Something."""
        y_0 = 0.69073
        y_1 = 7.5534
        return ((pi**2)*((1.0 + (y_0/eps))**(-8)) + y_1*(eps**(-8/7)))**(-1/8)
    
    def _dawn_temper(self,eps):
        """Something."""
        return (pi + (3.0*pi/eps)**(4/3))**(-1/4)
    
    
    def Approx_MaxDuskDawn_Temps(self,epsilon="self"):
        """Blah blah blah
        
        Some bozo.
        
        """
        if isinstance(epsilon,(float,int)):
            eps = epsilon
        else:
            eps = self.recirc_effic
        
        # Each gives you single T values.
        Ttilda_max = self._max_temper(eps)
        Ttilda_dusk = self._dusk_temper(eps)
        Ttilda_dawn = self._dawn_temper(eps)
        
        return Ttilda_max,Ttilda_dusk,Ttilda_dawn
