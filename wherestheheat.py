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

pi = np.pi


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
    
    def _calc_efactor(self,ecc):
        # For scaling ang. vel. at periastron (^-1 for period)
        return ((1-ecc)**(-1.5))*((1+ecc)**0.5)

    def _calc_orb_period(self):
        return 2.0*pi*(((self.smaxis**3.0)/(self.Mstar*self.grav_const))**0.5)
    
    def _kepler_calcs(self):
        """Calculates orbital separation (between planet and its star) as a function of time
            and saves it in __init_.
            Used in calculating incident flux.
            
            Note
            ----
            The fitter class calls this for an initial time array, changes the time array, then calls
            it's own radius function to get these quantities for an input time array.
            
            It works for now but it's buggy. Might want to review this interaction.
            
            
            Parameters
            ----------
            None
            
            Uses
            -------
            
            pyasl from PyAstronomy to calculate true anomaly
            
            Returns
            -------
            
            t (1D array)
            Time array in seconds, of length pmax * int(Porb/ 24hrs)*steps.
            
            radius (1D array)
            Same length as t. Orbital separation radius array (as a function of time)
            
            ang_vel (1D array)
            Orbital angular velocity as a function of time.
            
            alpha (1D array)
            Phase angle as a function of time.
            (90-self.argp) - np.array(TA)*57.2958
            
            f (1D array)
            Planet illuminated fraction
            0.5*(1-np.cos(alpha*pi/180.0))
            
        """
        # The KeplerEllipse coordinates are: x == "North", y == "East", z == "away from observer".
        # Our orbits are edge-on with inclination = 90 degrees, so orbits in x-z plane.
        # Longitude of ascending node doesn't really matter, so we set Omega = 0 degrees (along +x axis).
        # Argument of periastron measured from ascending node at 1st quarter pahse (alpha = 90 deg).
        # >>> SEE KEY NOTE IN _modify_arg_peri METHOD!!!
        ke = pyasl.KeplerEllipse(self.smaxis,self.Porb,e=self.eccen,Omega=0.0,w=self.arg_peri,i=90.0)
        
        # Periastron happens at t = 0 because "tau" in KeplerEllipse defaults to zero.
        # Get info for conjunctions--transit and eclipse--when orbit crosses y-z plane:
        conjunc_times = np.array(ke.yzCrossingTime())
        conjunc_pos,conjunc_tru_anom = ke.xyzPos(conjunc_times,getTA=True)
        # Eclipse has +z (transit -z) so argmax is eclipse index (0 or 1).
        i_ecl = np.argmax(conjunc_pos[:,2])
        
        self.ecl_time = conjunc_times[i_ecl]
        self.ecl_pos = conjunc_pos[i_ecl]
        self.ecl_tru_anom = conjunc_tru_anom[i_ecl]
        self.tr_time = conjunc_times[1-i_ecl]
        self.tr_pos = conjunc_pos[1-i_ecl]
        self.tr_tru_anom = conjunc_tru_anom[1-i_ecl]
        
        # Make time array
        tmax = self.Porb*self.numOrbs
        Nmin = int(self.numOrbs*self.stepsPerOrbit)
        if self.continueOrbit:
            timeval = np.linspace(self.again_t,self.again_t + tmax,num=Nmin+1)
        else:
            timeval = np.linspace(0,tmax,num=Nmin+1)
        
        radius = ke.radius(timeval)
        pos,tru_anom = ke.xyzPos(timeval,getTA=True)
        
        # Want alpha(transit) = 0 and alpha(periapsis) = 90 + arg_peri.
        # So: alpha = 90 + arg_peri + tru_anom
        alpha = (90.0 + self.arg_peri + np.degrees(np.array(tru_anom))) % 360.0
        # Minus here because alpha = 0 at transit.
        frac_litup = 0.5*(1.0 - np.cos(np.radians(alpha)))
        
        self.timeval = timeval
        self.radius = radius
        self.tru_anom = tru_anom
        self.alpha = alpha
        self.frac_litup = frac_litup
        return
    
    def _phases_SOpoints(self):
        """
            Calculates coordinates of substellar point wrt to the location of the
            substellar point at periastron (theta = Pi/2, phi =0).
            Used to rotate the coordinate array array as a function of time .
            
            Also calculates coordinates of subobserver location wrt subobserver location at periastron.
            
            Note
            ----
            DEPENDS ON WADV!! CAN'T be stored in __init__ . I tried.
            
            Parameters
            ----------
            None
            
            
            
            Returns
            -------
            
            t (1D array)
            Time array in seconds, of lenght pmax * int(Porb/ 24hrs)*steps.
            
            zt (1D array)
            Same lenght as t. Cumulative orbital angular displacement.
            
            SSP (1D array)
            Same lenght as t. SSP = ((zt mod (2 Pi/ Rotation)) mod (2 Pi/ orbit));
            Gives coordinate of substellar point relative
            to the substellar point at periastron (located at theta = Pi/2, phi = 0).
            
            
            SOP (1D array)
            Coordinates of sub-observer point mod (2Pi/Rotation). Only used for testing.
            
        """
        # Planet coordinates: longitude = 0 always points at star.
        # So, longitude of gas parcels change throughout orbit. Colatitude stays the same.
        # Gives: new_longs = orig_longs + Rotation effect (East) - Orbit effect (West)
        shift_longs = self.longs[np.newaxis,:] + ((2.0*pi/self.Prot)*self.timeval[:,np.newaxis]) - self.tru_anom[:,np.newaxis]
        longs_evolve = shift_longs % (2.0*pi)
        
        # Sub-stellar point: always long = 0 in our coordinates.
        SSP_long = 0

        # Sub-observer point: longitude from alpha (orbital phase for observer)
        # SOP_long = pi when alpha = 0, and Westward drift means -alpha.
        SOP_long = (pi - np.radians(self.alpha)) % (2.0*pi)

        self.longs_evolve = longs_evolve
        self.SSP_long = SSP_long
        self.SOP_long = SOP_long
        return
    
    def _phiT_visillum(self):
        """Creates coordinate matrix wrt substellar point at each point in time.
            Creates initial temperature array.
            Calculates weight function to be applied to stellar flux to obtain
            incident flux a each location on the planet, at each point in time.
            
            Note
            ----
            DEPENDS ON WADV; CAN'T BE STORED IN __init__
            In places where initial temperature is 0, we replace T = 0 with T = 0.1 to avoid overflows.
            At t = 0, the planet is at periastron and the substellar point
            is located at theta = Pi/2, phi = 0
            
            Parameters
            ----------
            None
            
            Calls
            -------
            
            self.SSP(pmax, steps) to get:
            
            t
            1D time array of lenght pmax * int(Porb/ 24hrs)*steps;
            in seconds
            
            SSP
            1D array, same lenght as t. Gives coordinate of substellar point relative
            to the substellar point at periastron (located at theta = Pi/2, phi = 0).
            Used to rotate the coordinate array array as a function of time .
            
            Returns
            -------
            
            d
            3D position and temperature array;
            
            shape = (len(time), NPIX, 3)
            
            d[:,:,0] = thetas (latitude -- 0 to Pi, equator at Pi/2)
            *remains constant as a function of time
            
            d[:,:,1] = phis (longitude -- 0 to 2Pi); phi(t) = phi(0)+SSP(t)
            
            
            d[:,:,2] = starting temperature array
            
            Fweight
            2-D array that represents the weight applied to the stellar flux
            at each location on the planet to obtain incident flux at each moment in time.
            --- shape is (lenght time, NPIX)
            
            coordsSOP (2D array)
            coordinates relative to the suborserver point
            
            weight (2D array )
            visibility array to be applied to flux array in coordinates wrt SSP
            (shape is (#time steps, NPIX))
            
        """
        if self.continueOrbit:
            Tvals = self.again_Tmap
        else:
            the_low_case = (0.5*(np.cos(self.longs) + np.absolute(np.cos(self.longs)))*np.sin(self.colat))**0.25
            the_high_case = (np.sin(self.colat)/pi)**0.25
            pos_eps = abs(self.epsilon)  # Negative epsilons don't play nice in "the_scaler".
            the_scaler = (pos_eps**1.652)/(1.828 + pos_eps**1.652)  # Estimated curly epsilon, Schwartz et al. 2017
            Tvals = the_scaler*the_high_case + (1.0 - the_scaler)*the_low_case  # E.B. model parameterization
            Tvals[Tvals<0.1] = 0.1
        
        Tvals_evolve = np.zeros(self.longs_evolve.shape)
        Tvals_evolve[0,:] += Tvals
        
        longs_minus_SSP = self.longs_evolve - self.SSP_long
        longs_minus_SOP = self.longs_evolve - self.SOP_long[:,np.newaxis]
        
        illumination = 0.5*(np.cos(longs_minus_SSP) + np.absolute(np.cos(longs_minus_SSP)))*np.sin(self.colat)
        visibility = 0.5*(np.cos(longs_minus_SOP) + np.absolute(np.cos(longs_minus_SOP)))*np.sin(self.colat)
        
        self.Tvals_evolve = Tvals_evolve
        self.illumination = illumination
        self.visibility = visibility
        return
    
    def _modify_arg_peri(self,ap):
        # KEY!>>>: Argument of periastron measured from ascending node at 1st quarter pahse (alpha = 90 deg).
        #     >>>: But in exoplanet literature, arg. peri. = 90 deg means periastron at TRANSIT (alpha = 0 deg).
        #     >>>: (FYI, the arg. peri. quoted in papers are probably for the host stars.)
        #     >>>: So, we add 180 deg to the input argument for consistency. DON'T GET CONFUSED! :-)
        return (ap+180.0) % 360.0
    
    
    def __init__(self,name='Hot Jupiter',Teff=5778,Rstar=1.0,Mstar=1.0,
                 Rplanet=1.0,smaxis=0.1,eccen=0,arg_peri=0,bondA=0,
                 motions='calc',orbval=1.0,rotval=1.0,
                 constant='radiate',conval=12.0,
                 numOrbs=3,stepsPerRot=360,NSIDE=8,
                 again_Tmap=False,again_t=0,continueOrbit=False):
        
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

        self.Teff = Teff  # star effective temp
        self.Rstar = Rstar*self.radius_sun  # star radius
        self.Mstar = Mstar*self.mass_sun # star mass
        self.Rplanet = Rplanet*self.radius_jupiter  # planet mass
        self.smaxis = smaxis*self.astro_unit  # semimajor axis
        self.eccen = eccen  # eccentricity
        # For scaling ang. vel. at periastron (^-1 for period)
        self._ecc_factor = self._calc_efactor(self,eccen)
        
        # KEY!>>>: Argument of periastron measured from ascending node at 1st quarter pahse (alpha = 90 deg).
        #     >>>: But in exoplanet literature, arg. peri. = 90 deg means periastron at TRANSIT (alpha = 0 deg).
        #     >>>: (FYI, the arg. peri. quoted in papers are probably for the host stars.)
        #     >>>: So, we add 180 deg to the input "arg_peri" for consistency. DON'T GET CONFUSED! :-)
        self.arg_peri = self._modify_arg_peri(self,arg_peri)
        
        self.bondA = bondA  # planet Bond albedo

        ### NEED TO PICK UP FROM HERE AND RELATED METHODS NEXT TIME ###
        if motions == 'calc':  # Periods calculated in seconds
            self.Porb = self._calc_orb_period()
            self.Wrot = rotval*((2.0*pi/self.Porb)*self._ecc_factor)
            rp = lambda w: np.inf if w == 0 else 2.0*pi/abs(w)
            self.Prot = rp(self.Wrot)
        elif motions == 'per':  # Periods converted from days to seconds
            self.Porb = orbval*self.sec_per_day
            self.Prot = abs(rotval)*(self.Porb/self._ecc_factor)
            rw = lambda p,v: 0 if p == np.inf else np.sign(v)*(2.0*pi/p)
            self.Wrot = rw(self.Prot,rotval)
        elif motions == 'freq':  # Ang. freq. converted from degrees/day to rad/second
            ov = np.radians(orbval)
            self.Porb = (2.0*pi/ov)*self.sec_per_day
            self.Wrot = rotval*((2.0*pi/self.Porb)*self._ecc_factor)
            rp = lambda w: np.inf if w == 0 else 2.0*pi/abs(w)
            self.Prot = rp(self.Wrot)
        self.adv_freq_peri = self.Wrot - ((2.0*pi/self.Porb)*self._ecc_factor)

        self.Tirrad = self.Teff*((1-self.bondA)**0.25)*((self.Rstar/(self.smaxis*(1-self.eccen)))**0.5)

        # Handling tau_rad and epsilon, depending on inputs
        if constant == 'radiate':
            self.tau_rad = conval*self.sec_per_hour  # tau_rad converted from hours to seconds
            self.epsilon = self.adv_freq*self.tau_rad  # epsilon has same sign as adv_freq
        elif constant == 'recirc':
            sn = lambda w: -1.0 if w < 0 else 1.0
            self.epsilon = abs(conval)*sn(self.adv_freq)  # Negative for net-Westward circulation
            if self.adv_freq == 0:
                self.tau_rad = 0
            else:
                self.tau_rad = abs(self.epsilon/self.adv_freq)  # tau_rad in seconds
        
        # PRE-CALCULATED FUNCTIONS - Some of this stuff can be edited or removed.
        self.rotationsPerOrbit = np.ceil(max(self.Porb/self.Prot,1))  # For the default time length
        self.numOrbs = numOrbs  # Num. of orb. periods
        self.stepsPerOrbit = stepsPerRot*self.rotationsPerOrbit  # Steps per orbit
        
        ### JCS Things - want to make these automatic eventually.
        self.again_Tmap = again_Tmap
        self.again_t = again_t
        self.continueOrbit = continueOrbit

        self._kepler_calcs()
        
        self.NSIDE = NSIDE
        self.colat,self.longs = hp.pix2ang(self.NSIDE,list(range(hp.nside2npix(self.NSIDE))))
        
        self._phases_SOpoints()
        self._phiT_visillum()
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
        
        # adv_freq, tau_rad, epsilon
        form_cols = '{:^26} {:^22} {:^10}'
        print(form_cols.format('Advective freq. (rad/hr)','Radiative time (hrs)','Epsilon'))
        form_cols = '{:^26.3f} {:^22.3f} {:^10.3f}'
        print(form_cols.format(self.adv_freq*self.sec_per_hour,self.tau_rad/self.sec_per_hour,self.epsilon))
        
        return


     
    """ FUNCTIONS FOR DEFINING THE PLANET CONDITIONS, STELLAR FLUX """

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
            if (abs(self.epsilon) <= 10**(-4)) or (self.tau_rad <= 10**(-4)):
                self.Tvals_evolve = ((1.0-self.bondA)*self.illumination)**(0.25)
            else:
                sn = lambda w: -1.0 if w < 0 else 1.0
                delta_longs = (self.longs_evolve[1:,:] - self.longs_evolve[:-1,:]) % (sn(self.adv_freq)*2.0*pi)
                
                for i in range(1,len(self.timeval)):
                    # Stellar flux is constant for circular orbits, F(t)/Fmax = 1.
                    delta_Tvals = (1.0/self.epsilon)*(self.illumination[i-1,:] - (self.Tvals_evolve[i-1,:]**4))*delta_longs[i-1,:]
                    self.Tvals_evolve[i,:] = self.Tvals_evolve[i-1,:] + delta_Tvals  # Step-by-step T update
        else:
            # Normalized stellar flux, can clean up becasue A LOT cancels.
#            flux_inc = self.Finc()[:,np.newaxis]
#            flux_max = self.stef_boltz*(self.Teff**4)*((self.Rstar/(self.smaxis*(1.0-self.eccen)))**2)
#            scaled_illum = (flux_inc/flux_max)*self.illumination
            scaled_illum = self.illumination*((self.smaxis*(1-self.eccen)/self.radius[:,np.newaxis])**2)
            
            # Eccentric DE uses t_tilda = t/tau_rad
            if (abs(self.epsilon) <= 10**(-4)) or (self.tau_rad <= 10**(-4)):
                self.Tvals_evolve = ((1.0-self.bondA)*scaled_illum)**(0.25)  # Why divided by stef_boltz before??
            else:
                delta_radtime = (self.timeval[1:] - self.timeval[:-1])/self.tau_rad
                
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
            eps = self.epsilon
        
        # Each gives you single T values.
        Ttilda_max = self._max_temper(eps)
        Ttilda_dusk = self._dusk_temper(eps)
        Ttilda_dawn = self._dawn_temper(eps)
        
        return Ttilda_max,Ttilda_dusk,Ttilda_dawn
