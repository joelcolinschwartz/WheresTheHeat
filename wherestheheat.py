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
    
    radius_sun = (6.957)*(10**8)  # in m
    radius_jupiter = (6.9911)*(10**7)  # in m
    mass_sun = (1.98855)*(10**30)  # in kg
    mass_jupiter = (1.8982)*(10**27)  # in kg
    grav_const = (6.67408)*(10**(-11))  # Gravitational constant
    
    planck = (6.62607004)*(10**(-34))
    speed_light = (2.99792458)*(10**8)  # in m/s


    def _period_calc(self):
        self.Porb = 2.0*pi*(((self.smaxis**3.0)/(self.Mstar*self.grav_const))**0.5)
        self.Prot = self.Porb*((1-self.eccen)**1.5)*((1+self.eccen)**(-0.5))
        return
    
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
        ke = pyasl.KeplerEllipse(self.smaxis,self.Porb,e=self.eccen,Omega=180.0,i=90.0,w=self.arg_peri)
        
        # Make time array
        tmax = self.Porb*self.numOrbs
        Nmin = int(self.numOrbs*self.stepsPerRot)
        if self.continueOrbit:
            timeval = np.linspace(self.again_t,self.again_t + tmax,num=Nmin+1)
        else:
            timeval = np.linspace(0,tmax,num=Nmin+1)
        
        radius = ke.radius(timeval)
        vel = ke.xyzVel(timeval)
        abs_vel = (vel[:,0]**2.0 + vel[:,1]**2.0 + vel[:,2]**2.0)**0.5
        ang_vel = abs_vel/radius
        
        # TA == true anomaly
        pos,TA = ke.xyzPos(timeval,getTA=True)
        
        # Want 0 at transit and TA = 90-w at transit, so alpha = 90-w - TA
        alpha = (90.0-self.arg_peri) - np.degrees(np.array(TA))
        frac_litup = 0.5*(1.0 - np.cos(np.radians(alpha)))
            
        self.timeval = timeval
        self.radius = radius
        self.ang_vel = ang_vel
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
        orbital_phase = np.zeros(self.timeval.shape)
        if self.continueOrbit:
            orb_pos_zero = (self.again_t*(2.0*pi/self.Porb)) % (2.0*pi)
        else:
            orb_pos_zero = 0
        orb_phase[0] = orb_pos_zero
        
        delta_t = self.timeval[1:] - self.timeval[:-1]
        orb_phase[1:] = orb_pos_zero + np.cumsum(self.ang_vel[:-1]*delta_t)
        
        # Using w_rot because w_adv includes w_orb- Yes?
        SSP = (orb_phase - ((2.0*pi/self.Prot)*self.timeval)) % (2.0*pi)
        # To get anti-SSP at t=0, do (-w_rot * t) + pi
        SOP = (pi - ((2.0*pi/self.Prot)*self.timeval)) % (2.0*pi)
        
        self.orb_phase = orb_phase
        self.SSP = SSP
        self.SOP = SOP
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
            the_low_case = (np.sin(self.thetas)*(np.cos(self.phis) + np.absolute(np.cos(self.phis)))/2.0)**0.25  # Need **0.25 outside EVERYTHING
            the_high_case = (np.sin(self.thetas)/pi)**0.25
            the_scaler = (self.epsilon**1.652)/(1.828 + self.epsilon**1.652)  # Estimated curly epsilon, Schwartz et al. 2017
            Tvals = the_scaler*the_high_case + (1.0 - the_scaler)*the_low_case  # E.B. model parameterization
            Tvals[Tvals<0.1] = 0.1
        
        # Original array has shape (time steps, num. of pixels, 3) where 3 is lat-long-temp.
        # Try to simplify: lats never change, keep longs and temps apart.
        phis_evolve = self.phis[np.newaxis,:] + (self.SSP[:,np.newaxis]*np.sign(self.wadv))
        Tvals_evolve = np.zeros(phis_evolve.shape)
        Tvals_evolve[0,:] += Tvals
        phis_relSOP = self.phis_evolve - np.radians(180.0-self.alpha[:,np.newaxis]) # For making visibility
        
        # Called "Fweight" before.
        illumination = 0.5*(np.cos(phis_evolve) + np.absolute(np.cos(phis_evolve)))*np.sin(self.thetas)
        visibility = 0.5*(np.cos(phis_relSOP) + np.absolute(np.cos(phis_relSOP)))*np.sin(self.thetas)
        
        self.phis_evolve = phis_evolve
        self.Tvals_evolve = Tvals_evolve
        self.illumination = illumination
        self.visibility = visibility
        return
    
    
    def __init__(self,name='Hot Jupiter',Teff=5778,Rstar=1.0,Mstar=1.0,
                 Rplanet=1.0,smaxis=0.1,eccen=0,arg_peri=0,bondA=0,
                 motions='calc',orbval=1.0,rotval=1.0,
                 tau_rad=12.0,epsilon=1.0,
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
        self.arg_peri = arg_peri  # angle betwen periastron and transit (degrees)
        
        self.bondA = bondA  # planet Bond albedo

        if motions == 'calc':
            self._period_calc()
            self.wadv = (2.0*pi/self.Prot) - (2.0*pi/self.Porb)
        elif motions == 'per':
            self.Porb = orbval*sec_per_day
            self.Prot = rotval*sec_per_day
            self.wadv = (2.0*pi/self.Prot) - (2.0*pi/self.Porb)
        elif motions == 'freq':  ## DO THESE NEED 'sec_per_day' MULTIPLIED?
            self.Porb = (2.0*pi/orbval)
            self.Prot = (2.0*pi/rotval)
            self.wadv = rotval - orbval

        self.T0 = self.Teff*((1-self.bondA)**0.25)*((self.Rstar/(self.smaxis*(1-self.eccen)))**0.5)

        if self.eccen == 0:
            self.epsilon = epsilon
            if self.wadv == 0:
                self.tau_rad = 0
            else:
                self.tau_rad = abs(epsilon/self.wadv)
        else:
            self.tau_rad = tau_rad*3600.0  # Because input tau_rad is in hours(?)
            self.epsilon = abs(self.wadv)*tau_rad
        
        # PRE-CALCULATED FUNCTIONS - Some of this stuff can be edited or removed.
        self.rotationsPerOrbit = np.ceil(max(self.Porb/self.Prot,1))  # For the default time length
        self.numOrbs = numOrbs  # Num. of orb. periods
        self.stepsPerRot = stepsPerRot*self.rotationsPerOrbit  # Steps per orbit
        
        ### JCS Things - want to make these automatic eventually.
        self.again_Tmap = again_Tmap
        self.again_t = again_t
        self.continueOrbit = continueOrbit

        self._kepler_calcs()
        
        self.NSIDE = NSIDE
        self.thetas,self.phis = hp.pix2ang(self.NSIDE,list(range(hp.nside2npix(self.NSIDE))))
        
        self._phases_SOpoints()
        self._phiT_visillum()
        return
        
        #####  #####
        
##        if Porb <= 0.0:
##            self.Porb = 2*pi*(self.a**3/(self.Mstar*parcel.grav_const))**(0.5)
##        else:
##            self.Porb =  Porb* self.sec_per_day #orbital period in seconds formula --
##            
##        ###JCS### self.P =  self.Prot() #self.Prot() P* parcel.sec_per_day#period of rotation planet in seconds
##        if wadv == -1:
##            self.P = np.inf
##        else:
##            self.P = self.Porb/(wadv + 1)  ### JCS: Rot. period less than orb. period by this factor.
##        
##        ### JCS 12/7/16: SOMETHING SEEMS OFF ABOUT EITHER THIS OR THE WAY wadv IS DEFINED
##        ### For 2nd, think: if e = 0 then Prot = Porb  here --> tidally locked
##        ### Written above that wadv = 1 means gas doesn't move w.r.t. sub-stellar point
##        ### But C+A 2011a has tidally locked atmosphere with wadv = 0.
##        ### Hmmm...
##        ###
##        ### AH! KEY POINT: Looks like you can't (well, rather shouldn't) specify Prot, Porb, AND ALSO wadv.
##        ### That's because wadv = wrot - worb
##        ### Unless I'm mixing up core and atmosphere angular velocities, this is likely the problem.
##        ### PICK UP FROM HERE NEXT TIME!!
##        
##        ###JCS### self.wadv = (2*pi/self.P)*wadv #think this is a param we need to fit for. In units of how much faster 
##                                        #the parcel moves compared to the period of the planet. 
##                                        #+ moves with rotation of planet, - against
##        self.wadv = ((2.0*pi/self.P) - (2.0*pi/self.Porb))  ### JCS: Now properly defining w_adv from rot. and orb. ang. velocities
##        
##        ###
##        
##        self.T0 = Teff*(1-A)**(0.25)*(self.Rstar/(self.a*(1-self.e)))**(0.5) # * np.sin(theta))**0.25 when the parcel 
##                                                                        #lives away from the equator
##        #self.epsilon = self.wadv*self.tau_rad
##        if epsilon is None:        
##            self.tau_rad = tau_rad * 3600.0 #it was in hours now its in seconds
##            self.epsilon = self.tau_rad * np.abs(self.wadv)
##            
##        else:
##            if self.wadv == 0:
##                self.epsilon = 0
##                self.tau_rad = 0
##            else:
##                self.epsilon = epsilon
##                self.tau_rad = np.abs(self.epsilon/self.wadv)
#
#        # we're not fitting, keeping time aray to default
#        self.NSIDE = NSIDE
#
#        #POSITIONS
#        coords =np.empty(((hp.nside2npix(self.NSIDE)),2))
#        for i in range(hp.nside2npix(self.NSIDE)):
#            coords[i,:]=(np.array(hp.pix2ang(self.NSIDE, i)))
#        self.phis = coords[:,1]
#        self.thetas = coords[:,0]

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
        print(form_cols.format(self.Rplanet/self.radius_jupiter,self.bondA,self.smaxis/self.astro_unit,self.T0))
        print('')
        
        # Porb, Prot, eccen, argp
        form_cols = '{:^14} {:^14} {:^14} {:^24}'
        print(form_cols.format('P_orb (days)','P_rot (days)','Eccentricity','Arg. periastron (deg.)'))
        form_cols = '{:^14.2f} {:^14.2f} {:^14.3f} {:^24.1f}'
        print(form_cols.format(self.Porb/self.sec_per_day,self.Prot/self.sec_per_day,self.eccen,self.arg_peri))
        print('')
        
        # wadv, tau_rad, epsilon
        form_cols = '{:^16} {:^22} {:^10}'
        print(form_cols.format('Advective freq.','Radiative time (hrs)','Epsilon'))
        form_cols = '{:^16.3f} {:^22.3f} {:^10.3f}'
        print(form_cols.format(self.wadv,self.tau_rad/3600.0,self.epsilon))
        
        return
    
    
#    def print_stuff(self):
#            """Simply prints the planetary characteristics that you assigned to your object,
#            as well as the ones that were calculated from the same information; Takes no arguments,
#            returns nothing. """
#
#            print("""-Name  = {0} (Name of model)
#                    -Teff = {1} K (Temperature Star)
#                    -Rstar = {2} R-sun (Radius Star in solar radii)
#                    -Rplanet = {3} R-sun (Radius Planet in solar radii)
#                    -a = {4} AU (semimajor axis)
#                    -e = {5} (eccentricity)
#                    -argp = {6} #angle betwen periatron and transit in degrees
#                    -A = {7} (Bond Albedo)
#                    -P = {8} days (period of rotation of planet) **might need fixing
#                    -Porb = {9} days (orbital period of planet)
#                    -wadv = {10} - units of angular frequency 2Pi/planetperiod *wadv??
#                    -T0 = {11} K (Temperature of substellar point at periastron)
#                    -tau_rad = {12} hrs (radiative time scale)
#                    -epsilon = {13} dimensionless circulation efficeincy param
#                    default rotationsPerOrbit = {14} #used for giving the default time lenght for DE
#                    **** all of these are converted to SI units but they need
#                    to be entered in the units mentioned here ***
#                    """.format (self.name, self.Teff, self.Rstar/self.radius_sun, self.Rplanet/self.radius_sun,
#                                self.smaxis/self.astro_unit, self.eccen, self.arg_peri,
#                                self.bondA,
#                                self.Prot/self.sec_per_day, self.Porb/self.sec_per_day,
#                                self.wadv*self.Prot/ (2*pi),
#                                self.T0, self.tau_rad/ 3600.0,
#                                self.epsilon, self.rotationsPerOrbit))

     
    """ FUNCTIONS FOR DEFINING THE PLANET CONDITIONS, STELLAR FLUX """
       
##    def Prot(self):
##
##        """Calculates rotational period :: Prot :: of the planet when given orbital period.
##        
##        Used only by __init__ . 
##        
##        Note
##        ----
##             wrot = 2*Pi/Prot is chosen to match wmax (orbital angular velocity at periastron );
##             wadv is expressed as a multiple of wmax, with ( - ) meaning a rotation in the oposite direction. 
##             
##    
##        Parameters 
##        ----------
##            None. Uses eccentricity and orbital period, from the planet's description. 
##        
##        Returns
##        -------
##            
##            Prot in seconds.
##
##            """
##        Prot = self.Porb*(1-self.e)**(1.5)*(1+self.e)**(-0.5)      
##        return Prot
    

    
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
        
#    def Finc_hemi(self):
#
#        """Energy incident on the lighted hemisphere of planet. Used for nothing, actually.
#
#
#        Parameters
#        ----------
#            None
#
#        Calls
#        -------
#
#            self.Finc(pmax, steps) to get flux incident on the substellar point.
#
#        Returns
#        -------
#
#                Finc_hemi (1D array)
#                    Theoretical total flux incident on planet at each moment on time;
#                    lenght pmax * int(Porb/ 24hrs)*steps.
#
#            """
#
#        # Einc = Finc* integral(phi: 0->2pi, theta: 0->pi/2) rp^2*cos(theta)sin(theta) dtheta dphi
#        # ---- integral overplanet coordinates
#        return self.Finc()*pi*self.Rplanet**2

    
    
    """FUNCTIONS FOR DEFINING THE COORDINATE SYSTEM AND MOVEMENT AS A FUNCTION OF TIME """
    """::: they all use module pyasl from PyAstronomy ::: """
    
#    def SSP(self):
#
#        """
#
#        Calculates coordinates of substellar point wrt to the location of the
#        substellar point at periastron (theta = Pi/2, phi =0).
#        Used to rotate the coordinate array array as a function of time .
#
#        Also calculates coordinates of subobserver location wrt subobserver location at periastron.
#
#        Note
#        ----
#             DEPENDS ON WADV!! CAN'T be stored in __init__ . I tried.
#
#        Parameters
#        ----------
#            None
#
#
#
#        Returns
#        -------
#
#                t (1D array)
#                    Time array in seconds, of lenght pmax * int(Porb/ 24hrs)*steps.
#
#                zt (1D array)
#                    Same lenght as t. Cumulative orbital angular displacement.
#
#                SSP (1D array)
#                    Same lenght as t. SSP = ((zt mod (2 Pi/ Rotation)) mod (2 Pi/ orbit));
#                    Gives coordinate of substellar point relative
#                    to the substellar point at periastron (located at theta = Pi/2, phi = 0).
#
#
#                SOP (1D array)
#                    Coordinates of sub-observer point mod (2Pi/Rotation). Only used for testing.
#
#            """
#
#
#
#        t = self.t
#        'z0= orbital position at t = 0'
#        #z0=self.arg_peri*pi/180
#        if self.continueOrbit == True:
#            z0 = (self.again_t*(2.0*pi/self.Porb)) % (2.0*pi)
#        else:
#            z0=0
#        'orbital phase'
#        zt = np.empty(len(self.t))
#        deltat = t[1::]-t[:-1:]
#        #deltat = (self.Prot*1.0)/(stepsi*1.0)
#        zt[0]=z0
#        ###JCS  for i in range(1,len(self.t)):
#        ###JCS      zt[i]= zt[i-1]+self.ang_vel[i-1]*deltat[i-1]
#
#        zt[1:] = zt[0] + np.cumsum(self.ang_vel[:-1]*deltat)  ###JCS: At least avoids 1 for-loop
#
#        #added the mod 2pi removed the rest of the trying to mod out by pi stuff
#        ###JCS### SSP =(zt-((self.wadv)*t))%(2*pi)# -((self.wadv*t)/(2*pi)).astype(int)*2*pi)-
#        #(t/self.Porb).astype(int)*2*pi)
#
#        SSP = (zt - ((2.0*pi/self.Prot)*t)) % (2.0*pi)  ###JCS: Trying w_rot instead because w_adv already includes w_orb in it.
#
#        ###JCS SOP = (((-self.alpha[0]+180)*pi/180)-
#        ###JCS         ((self.wadv)*t)%(2*pi))#-((self.wadv*t)/(2*pi)).astype(int)*2*pi))
#
#        SOP = (pi - ((2.0*pi/self.Prot)*t)) % (2.0*pi)  ###JCS: (-w_rot * t) + pi, to get anti-SSP at t=0
#
#        #SOP = SSP + ((-alpha+180)*pi/180)
#        return t, zt, SSP, SOP


    """ILLUMINATION AND VISIBILITY """    
    
#    def illum(self) :
#
#        """Creates coordinate matrix wrt substellar point at each point in time.
#        Creates initial temperature array.
#        Calculates weight function to be applied to stellar flux to obtain
#        incident flux a each location on the planet, at each point in time.
#
#        Note
#        ----
#            DEPENDS ON WADV; CAN'T BE STORED IN __init__
#            In places where initial temperature is 0, we replace T = 0 with T = 0.1 to avoid overflows.
#            At t = 0, the planet is at periastron and the substellar point
#            is located at theta = Pi/2, phi = 0
#
#        Parameters
#        ----------
#            None
#
#        Calls
#        -------
#
#            self.SSP(pmax, steps) to get:
#
#                t
#                    1D time array of lenght pmax * int(Porb/ 24hrs)*steps;
#                    in seconds
#
#                SSP
#                    1D array, same lenght as t. Gives coordinate of substellar point relative
#                    to the substellar point at periastron (located at theta = Pi/2, phi = 0).
#                    Used to rotate the coordinate array array as a function of time .
#
#        Returns
#        -------
#
#                d
#                    3D position and temperature array;
#
#                    shape = (len(time), NPIX, 3)
#
#                    d[:,:,0] = thetas (latitude -- 0 to Pi, equator at Pi/2)
#                    *remains constant as a function of time
#
#                    d[:,:,1] = phis (longitude -- 0 to 2Pi); phi(t) = phi(0)+SSP(t)
#
#
#                    d[:,:,2] = starting temperature array
#
#                Fweight
#                    2-D array that represents the weight applied to the stellar flux
#                    at each location on the planet to obtain incident flux at each moment in time.
#                    --- shape is (lenght time, NPIX)
#
#
#            """
#
#        #TIME
#        Nmin = len(self.t) #int((pmaxi)*stepsi)
#        #tmax = self.Prot*pmaxi
#        t = self.t
#
#
#        phis = self.phis#.copy()
#        thetas = self.thetas#.copy()
#        ###JCS### Trying alternate Start_T below (nothing wrong with these)
###        #STARTING TEMPERATURES
###        if self.epsilon >= 20:
###            T = 0.75*(np.sin(thetas)**0.25)*(np.cos(phis)+np.abs(np.cos(phis)))/2
###            T[np.where(T<0.1)]=0.1
###        else:
###            T = (np.sin(thetas)**0.25)*(np.cos(phis)+np.abs(np.cos(phis)))/2
###            T[np.where(T<0.05)]=0.05
#
#        ###JCS Starting Temps
#        the_low_case = (np.sin(thetas)*(np.cos(phis) + np.absolute(np.cos(phis)))/2.0)**0.25  # Need **0.25 outside EVERYTHING
#        the_high_case = (np.sin(thetas)/pi)**0.25
#        the_scaler = (self.epsilon**1.5)/(1.8 + self.epsilon**1.5)  # Estimated curly epsilon
#        T = the_scaler*the_high_case + (1.0 - the_scaler)*the_low_case  # E.B. model parameterization
#        T[T<0.1] = 0.1
#
#        if self.continueOrbit == True:
#            T = self.again_Tmap
#
#        #3D ARRAY TO CONTAIN EVERYTHING. CALLED IT d
#        #c= np.array(zip(thetas,phis,T)).reshape(-1,hp.nside2npix(self.NSIDE),3)
#        c = np.transpose(np.vstack((thetas,phis,T))).reshape(-1,hp.nside2npix(self.NSIDE),3)
#        d = np.repeat(c,Nmin, axis = 0)
#
#
#
#        #CREATE THE COORDINATE ARRAY. THERE'S 2 CASES: CIRCULAR ORBIT AND ECCENTRIC ORBIT
#        if self.eccen == 0.0:
#            t, zt, SSP, SOP = self.SSP()
#            "don't need deltaphi here but its good to know what it is"
#            #deltaphi = (2*pi/self.stepsi)* self.wadv/(2*pi/self.Prot)
#            d[:,:,1]= (phis)+(SSP.reshape(-1,1))*(np.sign(self.wadv))
#
#            #+(deltaphi* np.array(range(0,Nmin)).reshape(-1,1)))%(2*pi)
#            #-((self.wadv)*t[i] - int(self.wadv*t[i])) #update location of gas parcel
#
#
#        else:
#            "don't need deltaphi here"
#            #deltaphi = (2*pi/stepsi)* self.wadv/(2*pi/self.Prot)
#            t, zt, SSP, SOP = self.SSP()
#            #d[:,:,1]= (phis)+ ((zt.reshape(-1,1))*(np.sign(self.wadv)))%(2*pi)
#
#            d[:,:,1]= (phis)+ (SSP.reshape(-1,1))*(np.sign(self.wadv))
#            #+deltaphi* np.array(range(0,Nmin)).reshape(-1,1))%(2*pi) #added this to the mix
#
#
#        #ILLUMINATION WEIGHT FUNCTION. WILL GET PASSED TO DE ALONG WITH D THE COORDINATE MATRIX AND THE STUPID TIME MATRIX
#        Fweight = ((np.cos(d[:,:,1])+ np.abs(np.cos(d[:,:,1])))/2.0 * np.sin(d[:,:,0]))
#
#
#
#        return d, Fweight

    
    
        
#    def visibility(self, d = None, TEST = False):
#
#        """Calculates the visibility of each gas parcel on the planet,
#        i.e. how much flux is recieved by an
#        observer from each location on the planet as a function of time.
#
#        Note
#        ----
#            DEPENDS ON WADV; CAN'T BE STORED IN INIT
#            weight = ((np.cos(coordsSOP)+np.abs(np.cos(coordsSOP)))/2.0)*np.sin(thetas)
#
#
#        Parameters
#        ----------
#
#            d (3D array, optional)
#                position and temperature array; is provided as an argument by self.Fobs;
#                if not provided, self.illum() will be called and the array d will be calculated
#
#                    shape = (len(time), NPIX, 3)
#
#                    d[:,:,0] = thetas (latitude -- 0 to Pi, equator at Pi/2)
#                    *remains constant as a function of time
#
#                    d[:,:,1] = phis (longitude -- 0 to 2Pi); phi(t) = phi(0)+SSP(t)
#
#                    d[:,:,2] = starting temperature array
#
#                    EX:
#                    d[458,234,1] means: position wrt to substellar point of parcel # 234 at timestep 458
#
#            TEST (bool)
#                Usually False
#
#        Calls
#        -------
#
#            if d is None, calls self.illum() to get d.
#
#
#        Returns
#        -------
#
#            if TEST = False
#
#                weight
#                    visibility array to be applied to flux array in coordinates wrt SSP
#
#            if TEST = True
#
#                d (3D array)
#                    coordinates wrt SSP and temperature aray
#
#                coordsSOP (2D array)
#                    coordinates relative to the suborserver point
#
#                weight (2D array )
#                    visibility array to be applied to flux array in coordinates wrt SSP
#                    (shape is (#time steps, NPIX))
#
#
#            """
#
#        #if d is None:
#        #    try :
#        #        d = self.d
#        #    except AttributeError:
#        #        d, Fweight = self.illum()
#
#        if d is None:
#            d, Fweight = self.illum()
#            try:
#                d = d[self.stitch_point::,:,:]
#
#
#            except AttributeError:
#
#                d = d
#
#        else:
#
#            d = d
#
#        #t, zt, SSP, SOP = self.SSP(pmax, steps)
###        t,zt,SSP,SOP = self.SSP()  ###JCS: Will need SOP here, naturally.
###        SOP_longs = np.transpose(np.tile(SOP,(d.shape[1],1)))  ###JCS: Upping to match 'phis' below
###        zt_longs = np.transpose(np.tile(zt,(d.shape[1],1)))  ###JCS: Upping to match 'phis' below
#
#        phis = d[:,:,1] # location of gas parcel on planet relative to SSP
#
#        thetas = d[:,:,0]
#
#
#        try:
#            alpha = self.alpha[self.stitch_point::]
#
#        except AttributeError:
#            alpha = self.alpha
#        #coordsSSP = (phis)
#        coordsSOP = phis - (180.0 - alpha.reshape(-1,1))*pi/180.0  ###JCS: This was already what I was trying to do just below- duh!
#
###        coordsSOP = phis - (zt_longs + pi)  ###JCS: Compare SSP ang. to orb. position (zt+pi=0+pi=pi at t=0 so want phi=pi, etc.)
#
#        #2#coordsSOP = phis+(zt%(2*pi)).reshape(-1,1)
#
#        #coordsSOP = phis+(zt).reshape(-1,1)+(-alpha[0]+180)*pi/180
#        #coordsSOP = phis - (SOP - SSP).reshape(-1,1)
#
#        weight = ((np.cos(coordsSOP)+np.abs(np.cos(coordsSOP)))/2.0)*np.sin(thetas)  ###JCS: Same as version you're familiar with.
#
#        "THIS USES RESULT FROM ILLUM... MAYBE THEY CAN BE TOGETHER???"
#        if TEST :
#            return d, coordsSOP, weight
#        else:
#            return weight


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
            if (self.epsilon <= 10**(-4)) or (self.tau_rad <= 10**(-4)):
                self.Tvals_evolve = ((1.0-self.bondA)*illumination)**(0.25)
            else:
                # This difference is mod -2*pi --> why the negative?
                delta_phis = np.absolute((self.phis_evolve[1:,:] - self.phis_evolve[:-1,:]) % (-2.0*pi))
                
                for i in range(1,len(self.timeval)):
                    # Stellar flux is constant for circular orbits, F(t)/Fmax = 1.
                    delta_Tvals = (1.0/self.epsilon)*(self.illumination[i-1,:] - (self.Tvals_evolve[i-1,:]**4))*delta_phis[i-1,:]
                    self.Tvals_evolve[i,:] = self.Tvals_evolve[i-1,:] + delta_Tvals  # Step-by-step T update
        else:
            # Normalized stellar flux, can clean up becasue A LOT cancels.
            flux_inc = self.Finc()[:,np.newaxis]
            flux_max = self.stef_boltz*(self.Teff**4)*((self.Rstar/(self.smaxis*(1.0-self.eccen)))**2)
            scaled_illum = (flux_inc/flux_max)*self.illumination
            
            # Eccentric DE uses t_tilda = t/tau_rad
            if (self.epsilon <= 10**(-4)) or (self.tau_rad <= 10**(-4)):
                self.Tvals_evolve = ((1.0-self.bondA)*scaled_illum)**(0.25)  # Why divided by stef_boltz before??
            else:
                delta_radtime = (self.timeval[1:] - self.timeval[:-1])/self.tau_rad
                
                for i in range(1,len(self.timeval)):
                    delta_Tvals = (scaled_illum[i-1,:] - (self.Tvals_evolve[i-1,:]**4))*delta_radtime[i-1]
                    self.Tvals_evolve[i,:] = self.Tvals_evolve[i-1,:] + delta_Tvals  # Step-by-step T update
                    
        return
        
        ### ### ###
        
#        #print "Starting DE"
#
#        t = self.timeval
#        #d = self.dstart
#        #Fweight  =self.Fweight
#
#        d, Fweight = self.illum()
#
#
#        if self.eccen == 0.0:
#
#
#                if (self.epsilon <= 0.0001) or (self.tau_rad <= 0.0001):
#
#                    #d[:,:,2] = (((1-self.bondA)*Fweight)/self.stef_boltz)**(0.25)
#                    d[:,:,2] = (((1-self.bondA)*Fweight))**(0.25)
#                else:
#                    'changed this to work with arbitrary time'
#                    #deltaphi = (2*pi/stepsi) * self.wadv/(2*pi/self.Prot) #(i don't think wadv is important in this case)
#                    deltaphi = np.abs(((d[1::,:,1]-d[0:-1:,:,1]))%(-2*pi))
#
#
#                    for i in range(1,len(t)):#phis.shape[2]
#
#                        #incoming flux is always the same for a circular orbit F(t)/Fmax = 1
#
#                        dT =1.0/self.epsilon*(Fweight[i-1]-(d[i-1,:,2])**4 )*(deltaphi[i-1,:]) #calculate change
#
#                        d[i,:,2]= d[i-1,:,2]+ dT #update temperature array
#
#
#                #toc = time.time()
#                #print ("Time this took is: " , str(toc-tic), "seconds")
#
#                try:
#                    return t[self.stitch_point::], d[self.stitch_point::,:,:]
#
#                except AttributeError:
#                    return t, d
#
#
#
#
#
#        else:
#
#
#                #self.stef_boltz*self.Teff**4*(self.Rstar/np.array(self.radius(pmax,steps)[1]))**2
#                'normalized flux -- (minimum radius/ radius(t))**2'
#                Fstar = (self.Finc().reshape(-1,1)) #*Fweight
#
#                F = Fstar/(self.stef_boltz*self.Teff**4*(self.Rstar/(self.smaxis*(1-self.eccen)))**2)*Fweight
#                #F = ((self.smaxis*(1-self.eccen)/self.radius(pmax, steps)[1])**2)
#
#                if (self.epsilon <= 0.0001) or (self.tau_rad <= 0.0001):
#
#                    d[:,:,2] = (((1-self.bondA)*F)/self.stef_boltz)**(0.25)
#
#                else:
#                    "deltat will have to be changed for use in fitting"
#                    #deltat = self.Prot/stepsi
#                    deltat = t[1::]-t[:-1:]
#
#                    deltat_ = deltat/self.tau_rad
#                    wrot = (2*pi/self.Prot)* self.wadv/(2*pi/self.Prot)
#                    deltaphi = wrot*deltat
#
#                    for i in range(1,len(t)):
#
#
#                            dT =(( F[i-1] - (d[i-1,:,2])**4 )* (deltat_)[i-1])
#
#                            d[i,:,2]= d[i-1,:,2]+dT #update temperature array
#
#
#
#                try:
#                    return t[self.stitch_point::], d[self.stitch_point::,:,:]
#
#                except AttributeError:
#                    return t, d

    
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
        
        bolo_blackbody = self.stef_boltz*((self.Tvals_evolve*self.T0)**4)
        wavelength = microns*(10**(-6))  # microns to meters
        xpon = self.planck*self.speed_light/(wavelength*self.boltz*(self.Tvals_evolve*self.T0))
        ## So bolo and wave units match, leading "pi" is from integral over solid angle.
        wave_blackbody = pi*(2.0*self.planck*(self.speed_light**2)/(wavelength**5))*(1.0/(np.exp(xpon) - 1.0))
        
        return bolo_blackbody,wave_blackbody
    
    
        ### ### ###
        
#
#        wv = wavelength*10**(-6)
#
#        t, d = self.DE()
#
#
#        'Sometimes this has an overflow problem'
#
#
#        a = (2*self.planck*self.speed_light**2/wv**5)
#
#        b = (self.planck*self.speed_light*10**6)/(wavelength*self.boltz*self.T0)
#
#        Fwv = a* 1/(np.expm1(b/np.array(d[:,:,2])))
#
#
#        "Get the flux"
#        dA = hp.nside2pixarea(self.NSIDE)*self.Rplanet**2
#
#        Fmap_wv = (Fwv *dA)#/Fwvstar
#
#        Ftotal_ = (self.stef_boltz * (d[:,:,2]*self.T0)**4)*dA
#
#
#
#
#
#
#        if MAP:
#            Fleavingwv = np.zeros(len(t))
#            Ftotal = np.zeros(len(t))
#            for i in range(len(t)):
#
#                Fleavingwv[i] = np.sum(Fmap_wv[i,:])
#                Ftotal[i] = np.sum(Ftotal_[i,:])
#            crap, Fmap_wvpix = self.shuffle(d.copy(), Fmap_wv)
#
#            return t, d, Fmap_wv, Fmap_wvpix, Fleavingwv, Ftotal
#
#        else:
#            return t, d, Fmap_wv

    
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
        
        
        ### ### ###
        
#        #tic = time.time()
#        #print ("Starting Fobs")
#
#        pmaxi = self.pmaxi
#        #stepsi = self.stepsi
#
#
#        if MAP:
#
#            'call the functions we need'
#            t, d, Fmap_wv, Fmap_wvpix,Fleavingwv, Ftotal = self.Fleaving( wavelength, MAP = True)
#
#            #weight = self.weight
#            weight = self.visibility(d)
#
#            crap, weightpix = self.shuffle(d, weight)
#
#
#            'start doing what this function does'
#
#            #UNCOMMENT IF YOU WANNA CHECK THAT THE SHUFFLING ISNT DESTROYING ANYTHING
#            Fmapwvobs = Fmap_wvpix*weightpix
#            #Fmapwvobs_check = Fmap_wv*weight
#            Fwv = np.empty(len(t))
#            #Fwv_check = np.empty(Nmin)
#            for i in range(len(t)):
#
#                Fwv[i] = np.sum(Fmapwvobs[i,:])
#                #Fwv_check[i] = np.sum(Fmapwvobs_check[i,:])
#
#            #print "DO we get the same flux before and after shuffling? "
#            #print (Fwv == Fwv_check)
#
#            if PRINT == True:
#
#                fluxwv = np.array(Fwv).reshape(-1,1)
#                np.savetxt('observedflux_planet'+str(self.steps)+'_steps_per_period_'+str(pmaxi)+'periods_'
#                           +str(self.NSIDE)+'_NSIDE.out',
#                           zip( t, fluxwv), header = "time(planetdays),time(s), outgoing flux, outgoing flux per wv")
#            #toc = time.time()
#            #print ('Done with Fobs')
#            #print ('Time Fobs took: ' + str(toc-tic) + 'seconds')
#
#            if PRINT == False:
#                return  t, d, Fmapwvobs, weight, weightpix, Fwv
#
#        else:
#            t, d, Fmap_wv = self.Fleaving(wavelength)
#            weight = self.visibility(d)
#            ###JCS: Modified below here
#
#            if BOLO == True:
#                dA = hp.nside2pixarea(self.NSIDE)*self.Rplanet**2
#                Ftotal_ = (self.stef_boltz * (d[:,:,2]*self.T0)**4)*dA
#                Fmap_total = Ftotal_*weight
#                Ft = np.sum(Fmap_total, axis = 1)
#
#                if HEMISUM == True:  ###JCS: This means 'do you also want to straight-total the flux from each hemisphere you see?'
#                    hemi = np.copy(weight)  #JCS: Take the visibility...
#                    hemi[hemi > 0] = 1.0  #JCS: ...and turn it into a simple 'seen/unseen' mask...
#                    Hmap_total = Ftotal_*hemi
#                    Hflux = np.sum(Hmap_total, axis = 1)  #JCS: ...so you get the total hemisphere flux WITHOUT THE OBSERVER BIAS
#                    return t,d,Ft,Hflux
#                else:
#                    return t, d, Ft
#
#            else:
#                Fmapwvobs = Fmap_wv*weight
#                Fwv = np.sum(Fmapwvobs, axis = 1)
#
#                if HEMISUM == True:
#                    hemi = np.copy(weight)
#                    hemi[hemi > 0] = 1.0
#                    Hmapwvobs = Fmap_wv*hemi
#                    Hwvflux = np.sum(Hmapwvobs, axis = 1)
#                    return t,d,Fwv,Hwvflux
#                else:
#                    return t, d, Fwv

    ### You know, I don't think this is needed at all.
#    def shuffle (self, d = None, quantity = None):
#        """ This function will take an array for a quantity as well as it's coordinates
#        and rearrange it so it will correspond to healpy pixel number.
#
#
#        Note
#        ----
#        Normally you would provide a quantity defined on different surface patches
#        of the planet and the matching coordinate array, and this function will rearrange it to
#        correspong to pixel number.
#
#        For testing purposes, if these are not provided, shuffle() with get the d array from the DE
#        and shuffle the temperature values.
#
#        Parameters
#        ----------
#        d
#            3D numpy array. see DE()
#        quantity
#            2D array [time, value that needs rearranging]
#
#
#        Calls
#        -------
#
#        self.DE(pmax, steps, NSIDE) if d and quantity is not provided.
#
#
#
#        Returns
#        -------
#
#        d
#            unchanged
#
#        quantity
#            2D array[time, values for pixel];
#            quantity rearranged for drawing on a hp.mollview map.
#
#        """
#
#        if (d is None) and (quantity is None):
#            t, d = self.DE()
#            dd = np.array(d.copy())
#            quantity = np.array(d[:,:,2].copy())
#
#        else:
#            dd = np.array(d.copy())
#
#        timesteps = len(dd[:,0,0])
#            #quantity = quantity
#
#        #order = (hp.ang2pix(NSIDE, d[:,:,0], d[:,:,1])).astype(int)
#        #print order.shape
#        #quantity = quantity[order]
#        order = np.zeros((timesteps, hp.nside2npix(self.NSIDE)))
#        for i in range(timesteps):
#            order[i,:] = (hp.ang2pix(self.NSIDE, dd[i,:,0], dd[i,:,1])).astype(int)
#
#            quantity[i,:] = quantity[i,order[i,:].astype(int)]
#
#        print("shuffled quantity" )
#        return d, quantity

    
    def Calc_MaxDuskDawn_Temps(self):
        """Something.
            
        Something else.
        
        """
        start_time = int(self.stepsPerRot*(self.numOrbs-1))
        
        # For now, each gives you values at every time step.
        # Can do an extra check to get a single value.
        max_args = np.argmax(self.Tvals_evolve[start_time:,:],axis=1)
        max_angles = hp.pix2ang(self.NSIDE,max_args)
        
        Ttilda_max = np.amax(self.Tvals_evolve[start_time:,:],axis=1)
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
    

#    def phi_max(self,eps):
#
#        """ Finds analytic approximation for location of  Max temperature on the planet
#        for a circular orbit. Location is expressed in local stellar time.
#
#
#        Note
#        ----
#        Used for testing. Only works for circular orbits.
#
#
#        Parameters
#        ----------
#
#        eps (float)
#            efficiency parameter
#
#        Returns
#        -------
#
#        phi (rads)
#            Longitude at which gas reaches maximum temperature on planet.
#
#
#        """
#
#        x0 = 2.9685
#        x1 = 7.0623
#        x2 = 1.1756
#        x3 = -0.2958
#        x4 = 0.1846
#        f = x0*(1.0+x1*eps**(-x2+(x3/(1.0+x4*eps))))**(-1.0)
#        return np.arctan(f)
#
#
#
#    def Tmax (self,eps):
#        """ Finds analytic approximation for Max temperature on the planet
#        for a circular orbit.
#
#
#        Note
#        ----
#        Used for testing. Only works for circular orbits.
#
#
#        Parameters
#        ----------
#
#        eps (float)
#            efficiency parameter
#
#        Returns
#        -------
#
#        Tmax
#            Theoretical max temperature on planet in T/T0.
#
#        """
#
#        return np.cos(self.phi_max(eps))**(0.25)
#
#
#    def Tdusk (self,eps):
#        """ Finds analytic approximation for temperature at dusk on the planet
#        for a circular orbit.
#
#
#        Note
#        ----
#        Used for testing. Only works for circular orbits.
#
#
#        Parameters
#        ----------
#
#        eps (float)
#            efficiency parameter
#
#        Returns
#        -------
#
#        Tdusk
#            Theoretical temperature at dusk on planet (T/T0.)
#
#        """
#
#        y0 = 0.69073
#        y1 = 7.5534
#        f = (pi**2*(1.0+y0/eps)**(-8.0) + y1*eps**(-8.0/7.0))**(-1.0/8.0)
#        return f
#
#    def Tdawn (self,eps):
#        """ Finds analytic approximation for temperature at dawn on the planet
#        for a circular orbit.
#
#
#        Note
#        ----
#        Used for testing. Only works for circular orbits.
#
#
#        Parameters
#        ----------
#
#        eps (float)
#            efficiency parameter
#
#        Returns
#        -------
#
#        Tdawn
#            Theoretical temperature at dawn on planet (T/T0.)
#
#        """
#
#        f = (pi + (3*pi/eps)**(4.0/3.0))**(-0.25)
#        return f

    
class fitter (parcel):
    ''' Subclass of parcel class. The purpose of this is to return flux values suitable for fitting data.
    The main difference is: it will accept an input time array and stitch it to the
    time array that parcel generates. The input time array would be the 
    time values that come with your observations. The DE will run for the specified number of 
    orbital periods (default is 3) in order to reach a steady state. 
    However, Fobs() will only return flux values that match your input time-array.
    
    Problem
    ------- 
    There is no guarantee that a steady state has indeed been reached after 3 orbital periods.
    You have to make a figure and check. 3 or 4 periods seem to work well if the circulation efficiency isn't 
    huge. THIS COULD BE FIXED.
    
    Note
    ----
    You can call all the parcel functions from a fitter object. 
    However, you should only call Fobs(), i havent tested the other ones. 
                
    '''
    

    
    def __init__(self,parcel,ts = np.linspace(-60000,260000,num = 1000), me = 'HotWaterEarth', 
                 Teff =6000.0, Rstar = 1.0, Mstar = 1.5, Rplanet = 1.0870, a = 0.05, 
                 e = 0.1, argp = 0, 
                 A = 0, Porb = -1,
                 wadv = 2, tau_rad = 20, epsilon = None, pmax = 3, steps = 300, NSIDE = 8):
                     
                     
        '''
        See parcel.__init__() for all values except ts (the time array).
        
            
        Note
        ----
        It seems awkward to have to redefine all of the attributes of my parcel object when i
        only want to only change one of them. It's my first time trying inheritance, so there might be a more elegant way
        to do it. 
        
        Also, i had to changed self.name to self.me or there was a problem.
        
        Parameters
        ----------
        parcel parameters (see parcel.__init__())

        ts (1D numpy array): 
            time values that go with observations.
            
            ***Important:
            
            UNITS: has to be in seconds.
        
            TRANSIT: has to contain a transit and the transit has to occur at t=0!!! What this means is
            that you have to fit/ guess transit time and use as an input (your time array - transit time).
        
            SIGN: Times before transit get negative values. Times after transit take on positive values.
        
        
        t, radius, ang_vel, alpha, f (stored orbital position arrays):
            
            all of these arrays get recalculated to reflect the change in the time array.
        '''
        
        self.me = me
        super(fitter, self).__init__(self, 
                 Teff =Teff, Rstar = Rstar, Mstar = Mstar, Rplanet = Rplanet, a = a, 
                 e = e, argp = argp, 
                 A = A, Porb = Porb,
                 wadv = wadv, tau_rad = tau_rad, epsilon = epsilon, pmax = pmax, steps = steps, NSIDE = NSIDE)
        
        self.ts = ts
        
        self.time = self.t


        self.get_time_array()
        
        t, radius, ang_vel, alpha, frac_litup = self.radiussF()
        self.radius = radius
        self.ang_vel = ang_vel
        self.alpha = alpha
        self.frac_litup = frac_litup
        

        

    def get_time_array(self):
        
        '''
        Called by __init__ to create a suitable time array.
        Does so by stitching the default time array and the input time array together 
        based on where the transit occurs. It defines the new altered time array and attaches 
        it to the object as an attribute (self.t). It also defines the position of the stitching point 
        and attaches it to the object as self.stitch_point. That point is important for slicing the 
        outputs of the parcel functions to correspond to your time input.
        
        Note
        ----
        This function is slightly wonky. Without that if statement, it gets called twice for some reason.
        That screws everything up. It works the way it is now, but looks weird.
        
        Parameters
        ----------
            None; just make sure you defined your object right and your 
            time array has the righ characteristics.  See __init__ docs to
            know what those are.
            
        Returns
        -------
            Nothing. It does however define self.t and self.stitch_point
        
        '''
        
        if self.time is self.t:
            
            times = self.ts #needs to have the transit at t = 0!!!
            print(times[0])            
            tr = np.where(self.frac_litup <0.0001)[0]
            #ec = np.where(abs(self.frac_litup-1) < 0.0001)[0]
            transit_t = (self.time[tr])[-1]
            print('transit_t is', transit_t)
            #eclipse_t = np.array(self.t[ec])
            deltaT = self.time[1]-self.time[0]
            stitch_point = np.where(((self.time - (transit_t - abs(times[0])))<= deltaT) & 
            ((self.time - (transit_t - abs(times[0])))>=0))[0]
            print(stitch_point)
            print('time for stitching is',self.time[stitch_point])       
            newTimes  = np.concatenate((self.time[:stitch_point:],self.time[stitch_point]+times+abs(times[0])),axis = 0)
            
            self.t = newTimes
            self.stitch_point = stitch_point
            #return newTimes
            
    def radiussF(self):

        """Calculates orbital separation (between planet and its star) as well as other
        orbital position quantities as a function of time. 
        They all get attached to the object by __init_.
        Used in calculating incident flux.

        Note
        ----
             Exactly like the parcel._kepler_calcs function but i had to redefine it for use in
             recalculating the orbital positions to reflect the updated time array.
             
    
        Parameters
        ----------
            None

        Uses
        -------
            
            pyasl from PyAstronomy to calculate true anomaly
            
           
        Returns
        -------
            
                t (1D array)
                    Time array in seconds, of lenght pmax * int(Porb/ 24hrs)*steps. 
                    
                radius (1D array)
                    Same lenght as t. Orbital separation radius array (as a function of time) 
                    
                ang_vel (1D array)
                    Orbital angular velocity as a function of time.
                    
                alpha (1D array)
                    Phase angle as a function of time.
                    (90-self.arg_peri) - np.array(TA)*57.2958
                    
                f (1D array)
                    Planet illuminated fraction
                    0.5*(1-np.cos(alpha*pi/180.0))

        """   
        
        
        ke = pyasl.KeplerEllipse(self.smaxis, self.Porb, e=self.eccen, Omega=180., i=90.0, w=self.arg_peri)

        # Get a time axis

        print('observed time')
        t = self.t[:]

        radius = ke.radius(t)
        
        vel = ke.xyzVel(t)
        absvel = np.sqrt(vel[:,0]**2+vel[:,1]**2+vel[:,2]**2)
        ang_vel = absvel/ radius
        
        pos, TA = ke.xyzPos(t,getTA = True)
        
        # i want this to be 0 at transit. TA = 90-w at transit so alpha = 90-w - TA
        alpha = (90-self.arg_peri) - np.degrees(np.array(TA))
        
        frac_litup = 0.5*(1.0 - np.cos(np.radians(alpha)))
        
        return t, radius, ang_vel, alpha, frac_litup
        
    
if __name__ == '__main__':
    
    '''This part makes a quick figure to check that Fobs() still works and 
    that the stitching is doing the right thing. 
    The thicker line on the figure represents the data points that match your input times.
    The thin line is the original output of parcel using the default times. 
    You can also look at this figure to figure out if you've integrated long enough 
    to reach a steady state. The amount of time it needs depends on tau_rad and wadv.
    You can also use it to experiment with how many steps per day your 
    DE needs to calculate and still put out a reasonably smooth line.
    
    PROBLEM : If you dont have enough steps (like... less then 100/24 hours), the stitching might 
    have trouble finding a good spot to put your time values and/or the flux calculation will
    overflow.    
    '''
    
    gas = parcel(tau_rad = 10, wadv = 5, pmax = 4, e=0.3, steps = 300)
    gas2 = fitter(gas, tau_rad = 10, wadv = 5, pmax = 4, e= 0.3, steps = 300)
    
    gas2.get_time_array()
    
    #t,d = gas2.DE()
    import matplotlib.pyplot as plt 
    
    t,d,Fwv = gas2.Fobs()
    Fstar, Fstarwv = gas2.Fstar()
    #plt.plot(t[::],d[::,368,2], linewidth = 3)
    #plt.plot(t[::],d[::,368,2], linewidth = 3)
    
    #plt.plot(t,d[:,368,2], linewidth = 3)
    '''
    plt.plot(t[::],Fwv[::]/ max(Fwv), linewidth = 3)
    plt.plot(t[::],Fwv[::]/max(Fwv), linewidth = 3)'''
    
    plt.plot(t,Fwv/Fstarwv, linewidth = 3, label = 'fitter: flux at specified times')
    
    t,d,Fwv = gas.Fobs()
    Fstar, Fstarwv = gas.Fstar()
    #plt.plot(t[gas2.stitch_point::],d[gas2.stitch_point::,368,2], linewidth = 3)
    #plt.plot(t[:gas2.stitch_point:],d[:gas2.stitch_point:,368,2], linewidth = 3)
    
    #plt.plot(t,d[:,368,2])
    
    #plt.plot(t[gas2.stitch_point::],Fwv[gas2.stitch_point::]/ max(Fwv), linewidth = 3)
    #plt.plot(t[:gas2.stitch_point:],Fwv[:gas2.stitch_point:]/max(Fwv), linewidth = 3)
    
    plt.plot(t,Fwv/Fstarwv, label = 'parcel: flux at default times')
    
    plt.legend(loc = 4)
    
    
    '''
    gas = parcel(name = "HD189733b",Teff = 5938, e=0,Porb = 2.21, a = 0.031, wadv = 11.0, 
                          epsilon = 6, argp = 90, Rstar = 0.781, Mstar = 0.846,
                          Rplanet = 0.838, pmax = 10, steps = 300, NSIDE = 8 )      
    
    t, d, Fwv = gas.Fobs(BOLO = False)
    import matplotlib.pyplot as plt 
    plt.plot(t,Fwv)
    '''
    
    
    #gas2.print_stuff()
    #gas2.ts
    
