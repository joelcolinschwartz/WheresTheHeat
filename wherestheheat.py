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

import copy
#import warnings
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
from PyAstronomy import pyasl

pi = np.pi

with np.load('planck_integrals_1to10000K_01to30um.npz') as data:
    preplancked_temperatures_ = data['temperatures']
    preplancked_wavelengths_ = data['wavelengths']
    preplancked_integrals_ = data['integrals']

def _make_value_indexer(values):
    """Assumes integers >= 0 only."""
    val_used = values.astype(int)
    # 0 to max used
    val_full = np.arange(val_used[-1] + 1)
    full_to_used_inds = np.argmin(np.absolute(val_full[:,np.newaxis] - val_used[np.newaxis,:]),axis=1)
    return full_to_used_inds

preplancked_waveN_ = len(preplancked_wavelengths_)
# For general method that converts values to indices
preplancked_temper_valueinds_ = _make_value_indexer(preplancked_temperatures_)

# Modified inferno colormap
inferno_mod_ = copy.copy(plt.cm.inferno)
inferno_mod_.set_under('w')

# For custom graticule to draw on planet
grat_xfactor_ = np.sin(np.radians([30,60,90,120,150]))
grat_ylevels_ = np.cos(np.radians([30,60,90,120,150]))
dummy_clats = np.linspace(0,pi,101)
grat_sinclats_,grat_cosclats_ = np.sin(dummy_clats),np.cos(dummy_clats)
del dummy_clats

# For styling plot markers
orbloc_styles_ = {'transit':['b','^',150,'Transit'],
                  'eclipse':['y','v',150,'Eclipse'],
                  'ascend':['g','<',150,'Ascending Node'],
                  'descend':['m','>',150,'Descending Node'],
                  'periast':['r','D',100,'Periastron'],
                  'apast':['c','s',100,'Apastron'],
                  'phase':[[0.8,0.8,0.8],'o',150,'Planet Phase'],
                  'star':['y',['o','--'],300,'Star']}


def RecircEfficiency_Convert(epsilon,kind='infinite'):
    "Go between 0-inf <--> 0-1 epsilon."
    a,b = 1.652,1.828
    old_epsilon = np.absolute(np.atleast_1d(epsilon))  # Converts positive epsilons
    
    if kind == 'infinite':
        new_epsilon = (old_epsilon**a)/(b + (old_epsilon**a))  # Estimated curly epsilon, Schwartz et al. 2017
    elif kind == 'unity':
        new_epsilon = np.zeros(old_epsilon.shape)
        
        high_check = (old_epsilon >= 1.0)
        new_epsilon[high_check] = np.inf
        
        good_check = np.logical_not(high_check)
        new_epsilon[good_check] = ((b*old_epsilon[good_check])/(1.0 - old_epsilon[good_check]))**(1.0/a)  # Inverse of above
    else:
        print('RecircEfficiency_Convert error: strings for *kind* are')
        print('\"infinite\" to convert 0-inf into 0-1, or')
        print('\"unity\" to convert 0-1 into 0-inf.')
        return

    return new_epsilon


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
    
    _accept_motions = ('perR','perA','freqR','freqA')
    _accept_begins = ('transit','eclipse','ascend','descend','periast','apast')
    
    
    def _check_single_updater(self,thing):
        """Blah blah blah."""
        return thing != '_no'
    
    def _setup_scaled_quants(self,Rstar,Mstar,Rplanet,smaxis):
        """Blah blah blah."""
        scl_Rstar = Rstar*self.radius_sun if self._check_single_updater(Rstar) else self.Rstar  # star radius
        scl_Mstar = Mstar*self.mass_sun if self._check_single_updater(Mstar) else self.Mstar  # star mass
        
        scl_Rplanet = Rplanet*self.radius_jupiter if self._check_single_updater(Rplanet) else self.Rplanet  # planet mass
        scl_smaxis = smaxis*self.astro_unit if self._check_single_updater(smaxis) else self.smaxis  # semimajor axis
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
    
    def _parse_motion(self,motions,calc_orb,orbval,rotval):
        """Blah blah blah."""
        mots_loc,calo_loc,orbv_loc,rotv_loc = motions,calc_orb,orbval,rotval
        cur = lambda v: print('    Current value is {:}.'.format(v))
        blank = '; leave blank to keep current: '
        ok = '    OK, keeping current value.'
        
        if mots_loc not in self._accept_motions:
            print('Constructor warning: strings for *motions* are '+str(self._accept_motions)+'.')
            cur(self._last_motions)
            while mots_loc not in self._accept_motions:
                mots_loc = input('    Enter a valid *motions* (no quotes)'+blank)
                if mots_loc == '':
                    print(ok)
                    mots_loc = self._last_motions
                    break
            print('')
        
        if calo_loc not in [True,False]:
            print('Constructor warning: *calc_orb* is boolean.')
            cur(self._last_calc_orb)
            while calo_loc not in [True,False]:
                s = input('    Enter a valid *calc_orb* [T/F]'+blank).capitalize()
                if s == '':
                    print(ok)
                    calo_loc = self._last_calc_orb
                    break
                calo_loc = True if s in ['T','True'] else (False if s in ['F','False'] else '_bad')
            print('')
        
        # Check if *calc_orb* is False...
        if not calo_loc:
            if not isinstance(orbv_loc,(float,int)):
                print('Constructor warning: *orbval* is a float or an integer.')
                cur(self._last_orbval)
                while not isinstance(orbv_loc,(float,int)):
                    o = input('    Enter a valid *orbval*'+blank)
                    if o == '':
                        print(ok)
                        orbv_loc = self._last_orbval
                        break
                    try:
                        orbv_loc = eval(o)
                    except:
                        print('    Cannot eval, try again.')
                print('')
        else:
            # ...otherwise *orbval* doesn't matter.
            orbv_loc = orbv_loc if isinstance(orbv_loc,(float,int)) else self._last_orbval
        
        if not isinstance(rotv_loc,(float,int,list)):
            print('Constructor warning: *rotval* is a float, an integer, or a list (see docs).')
            cur(self._last_rotval)
            while not isinstance(rotv_loc,(float,int,list)):
                r = input('    Enter a valid *rotval*'+blank)
                if r == '':
                    print(ok)
                    rotv_loc = self._last_rotval
                    break
                try:
                    rotv_loc = eval(r)
                except:
                    print('    Cannot eval, try again.')
            print('')

        return mots_loc,calo_loc,orbv_loc,rotv_loc
    

    def _setup_orb_motion(self,motions,calc_orb,orbval):
        """Blah blah blah."""
        mot_style = motions[:-1]
        
        if calc_orb:  # Calculated in seconds
            Porb = self._calc_orb_period()
        elif mot_style == 'freq':  # Converted from degrees/day to rad/second
            Porb = (2.0*pi/np.radians(orbval))*self.sec_per_day
        elif mot_style == 'per':  # Converted from days to seconds
            Porb = orbval*self.sec_per_day
        return Porb
    
    
    def _rotation_builder(self,rotval):
        """Blah blah blah."""
        t_norm = self.timeval_rot/self.Porb  # seconds to number of orbits
        
        RV_built = rotval[0]  # Constant term
        for rv in rotval[1:]:
            kind = rv[0]
            rvA = np.asarray(rv[1:])
            
            if rvA.ndim == 2:  # [[order,coeff,offset],[...],...]
                order = rvA[:,0,np.newaxis]
                coeff = rvA[:,1,np.newaxis]
                offset = rvA[:,2,np.newaxis]
            elif rvA.ndim == 1:  # [coeff1,coeff2,...]
                order = np.arange(1,len(rvA)+1)[:,np.newaxis]
                coeff = rvA[:,np.newaxis]
                offset = 0
    
            if kind == 'time':
                alter_RV = (t_norm - offset)**order
            elif kind == 'peri_sin':
                alter_RV = np.sin(order*(self.tru_anom - np.radians(offset)))
            elif kind == 'peri_cos':
                alter_RV = np.cos(order*(self.tru_anom - np.radians(offset)))
            elif kind == 'phase_sin':
                alter_RV = np.sin(order*(np.radians(self.alpha - offset)))
            elif kind == 'phase_cos':
                alter_RV = np.cos(order*(np.radians(self.alpha - offset)))
            
            RV_built += np.sum(coeff*alter_RV,axis=0)
            
        return RV_built
    
    def _setup_rot_motion(self,motions,rotval):
        """Blah blah blah."""
        mot_style,mot_qual = motions[:-1],motions[-1]

        if isinstance(rotval,list):  # Calc values from rotval nested lists
            RV_built,rv_varies = self._rotation_builder(rotval),True
        elif isinstance(rotval,(float,int)):
            RV_built,rv_varies = rotval,False
        
        if mot_style == 'freq':  # Converted from degrees/day to rad/second
            if mot_qual == 'A':
                Wrot = np.radians(RV_built)/self.sec_per_day
            elif mot_qual == 'R':
                Wrot = RV_built*((2.0*pi/self.Porb)*self._ecc_factor)
            
            make_p = lambda w: np.inf if w == 0 else 2.0*pi/abs(w)
            w_to_p = np.vectorize(make_p) if rv_varies else make_p
            Prot = w_to_p(Wrot)
        
        elif mot_style == 'per':  # Converted from days to seconds
            if mot_qual == 'A':
                Prot = abs(RV_built)*self.sec_per_day
            elif mot_qual == 'R':
                Prot = abs(RV_built)*(self.Porb/self._ecc_factor)
            
            make_w = lambda p,r: 0 if p == np.inf else np.sign(r)*(2.0*pi/p)
            p_to_w = np.vectorize(make_w) if rv_varies else make_w
            Wrot = p_to_w(Prot,RV_built)
        
        Wadvec = Wrot - self.Worb
        return Prot,Wrot,Wadvec
    
    
#    def _setup_motion(self,motions,calc_orb,orbval,rotval):
#        """Blah blah blah."""
#        mot_style,mot_qual = motions[:-1],motions[-1]
#        w_to_p = lambda w: np.inf if w == 0 else 2.0*pi/abs(w)
#        p_to_w = lambda p,r: 0 if p == np.inf else np.sign(r)*(2.0*pi/p)
#
#        if calc_orb:  # Calculated in seconds
#            Porb = self._calc_orb_period()
#
#        if mot_style == 'freq':  # Converted from degrees/day to rad/second
#            if not calc_orb:
#                Porb = (2.0*pi/np.radians(orbval))*self.sec_per_day
#
#            if mot_qual == 'A':
#                Wrot = np.radians(rotval)/self.sec_per_day
#            elif mot_qual == 'R':
#                Wrot = rotval*((2.0*pi/Porb)*self._ecc_factor)
#            Prot = w_to_p(Wrot)
#
#        elif mot_style == 'per':  # Converted from days to seconds
#            if not calc_orb:
#                Porb = orbval*self.sec_per_day
#
#            if mot_qual == 'A':
#                Prot = abs(rotval)*self.sec_per_day
#            elif mot_qual == 'R':
#                Prot = abs(rotval)*(Porb/self._ecc_factor)
#            Wrot = p_to_w(Prot,rotval)
#
#        adv_freq_peri = Wrot - ((2.0*pi/Porb)*self._ecc_factor)
#        return Porb,Prot,Wrot,adv_freq_peri


    def _setup_radiate_recirc(self,tau_rad,epsilon):
        """Blah blah blah."""
        can_set_check = (self.eccen == 0) and isinstance(self.Wrot,(float,int))
        
        if epsilon != None:
            
            if can_set_check:
                recirc_effic = epsilon
                
                if self.Wadvec == 0:
                    if recirc_effic != 0:
                        print('Constructor warning: atmosphere\'s advective frequency is 0, *recirc_effic* is not.')
                        print('    Your planet has no winds, but you want to transport heat.')
                        print('    I am setting radiative time to infinity, but your system is not self-consistent.')
                        print('')
                        radiate_time = np.inf
                    else:
                        radiate_time = 0
            
                else:
                    radiate_time = abs(recirc_effic/self.Wadvec)
                    # Check for mismatched wind direction
                    if abs(np.sign(recirc_effic)-np.sign(self.Wadvec)) == 2:
                        print('Constructor warning: atmosphere\'s advective frequency and *recirc_effic* have opposite signs.')
                        print('    Your planet\'s winds flow one way, but you want them flowing the other way.')
                        print('    Radiative time is defined, but your system is not self-consistent.')
                        print('')

            else:
                print('Constructor ignore: you can only set *recirc_effic* if the atmosphere\'s advective frequency is constant.')
                if self.eccen != 0:
                    print('    Your planet\'s orbit is not circular (orbital angular velocity varies).')
                if not isinstance(self.Wrot,(float,int)):
                    print('    Your planet\'s spin is not constant (rotational angular velocity varies).')
                print('')
                recirc_effic = np.nan
                radiate_time = tau_rad*self.sec_per_hour  # Converted from hours to seconds
            
        else:
            radiate_time = tau_rad*self.sec_per_hour  # Converted from hours to seconds
            recirc_effic = self.Wadvec*radiate_time if can_set_check else np.nan
        
        return radiate_time,recirc_effic


    def _setup_time_array(self,_makenew):
        """Blah blah blah."""
        if _makenew:
            t_start = 0
            n_start = 0
        else:
            t_i = -1 if self._has_T_evolved else 0
            t_start = self.timeval[t_i]
            n_start = self.trackorbs[t_i]
        
        t_end = t_start + self.Porb*self.numOrbs
        N = round(self.numOrbs*self.stepsPerOrb)
        timeval = np.linspace(t_start,t_end,num=N+1)
        
        n_end = n_start + self.numOrbs
        trackorbs = np.linspace(n_start,n_end,num=N+1)
        return timeval,trackorbs
    
    def _reset_rot_times(self,_makenew):
        """Blah blah blah."""
        if _makenew:
            spin_history,timeval_rot = 0,np.copy(self.timeval)
        else:
            t_i = -1 if self._has_T_evolved else 0
            spin_history = self._net_zero_long[t_i]
            timeval_rot = np.copy(self.timeval - self.timeval[0])
        return spin_history,timeval_rot
    
    
    def _setup_the_orbit(self):
        """Ha ha ha."""
        # The KeplerEllipse coordinates are: x == "North", y == "East", z == "away from observer".
        # Our orbits are edge-on with inclination = 90 degrees, so orbits in x-z plane.
        # Longitude of ascending node doesn't really matter, so we set Omega = 0 degrees (along +x axis).
        # Argument of periastron measured from ascending node at 1st quarter phase (alpha = 90 deg).
        # >>> SEE KEY NOTE IN _modify_arg_peri METHOD!!!
        return pyasl.KeplerEllipse(self.smaxis,self.Porb,e=self.eccen,Omega=0.0,w=self.arg_peri,i=90.0)
    
    def _match_phase_new_orbit(self):
        """Stuff and things."""
        # Get old orbital phase
        p_i = -1 if self._has_T_evolved else 0
        the_phase = self.alpha[p_i]
        
        # Get new orbital phases
        t_demo = np.linspace(0,self.Porb,(10*self.stepsPerOrb)+1)
        _ig,tru_anom = self.kep_E.xyzPos(t_demo,getTA=True)
        alpha = (90.0 + self.arg_peri + np.degrees(np.array(tru_anom))) % 360.0
        
        # Get new time of matched phase
        tp_i = np.argmax(np.cos(np.radians(alpha - the_phase)))
        t_match = t_demo[tp_i]
        new_timeval = self.timeval - self.timeval[0] + t_match
        
        # Update orbit count if needed
        n_i = -1 if self._should_add_orbtime else 0
        n_start = self.trackorbs[n_i]
        n_end = n_start + self.numOrbs
        N = round(self.numOrbs*self.stepsPerOrb)
        new_trackorbs = np.linspace(n_start,n_end,num=N+1)
        return new_timeval,new_trackorbs
    
    def _calc_orbit_props(self):
        """Stuff and things."""
        radius = self.kep_E.radius(self.timeval)
        orb_pos,tru_anom = self.kep_E.xyzPos(self.timeval,getTA=True)
        
        vel_comp = self.kep_E.xyzVel(self.timeval)
        velocity = (np.sum(vel_comp**2.0,axis=1))**0.5
        Worb = velocity[0]/radius[0] if self.eccen == 0 else velocity/radius
        
        # Want alpha(transit) = 0 and alpha(periapsis) = 90 + arg_peri.
        # So: alpha = 90 + arg_peri + tru_anom
        alpha = (90.0 + self.arg_peri + np.degrees(np.array(tru_anom))) % 360.0
        # Minus here because alpha = 0 at transit.
        frac_litup = 0.5*(1.0 - np.cos(np.radians(alpha)))
        
        return radius,orb_pos,tru_anom,Worb,alpha,frac_litup
    
    def _find_conjunctions_nodes(self,kind):
        """Blah blah blah."""
        # Periastron happens at t = 0 because "tau" in KeplerEllipse defaults to zero.
        # Conjunctions--transit and eclipse--are when orbit crosses y-z plane.
        # Nodes--ascending and descending--would be when orbit crosses x-y plane (no method).
        # ---> BUT xzCrossingTime gives correct times, I've tested! (Because y-coor varies a tiny bit?)
        if kind == 'c':  # Conjunctions
            prop_times = np.array(self.kep_E.yzCrossingTime())
        elif kind == 'n':  # Nodes
            prop_times = np.array(self.kep_E.xzCrossingTime())
        
        prop_pos,prop_tru_anom = self.kep_E.xyzPos(prop_times,getTA=True)
        # Eclipse has +z, transit -z; Ascending node has +x, descending -x
        coor = lambda k: 2 if k == 'c' else 0
        i_plus = np.argmax(prop_pos[:,coor(kind)])
        
        plus_time = prop_times[i_plus]
        plus_pos = prop_pos[i_plus]
        plus_tru_anom = prop_tru_anom[i_plus]
        minus_time = prop_times[1-i_plus]
        minus_pos = prop_pos[1-i_plus]
        minus_tru_anom = prop_tru_anom[1-i_plus]
        
        return plus_time,plus_pos,plus_tru_anom,minus_time,minus_pos,minus_tru_anom

    
    def _setup_colatlong(self,NSIDE):
        """Blah blah blah."""
        colat,longs = hp.pix2ang(NSIDE,list(range(hp.nside2npix(NSIDE))))
        pixel_sq_rad = hp.nside2pixarea(NSIDE)
        on_equator = (colat == (pi/2))
        return colat,longs,pixel_sq_rad,on_equator,NSIDE
    
    ### PICK UP HERE TOO, THIS METHOD NEEDS EDITS WITH Wadvec ###
    def _calc_longs(self):
        """Blah blah blah."""
        # Planet coordinates: longitude = 0 always points at star.
        # Longitude of gas parcels change throughout orbit. Colatitude stays the same.
        # So: new_longs = orig_longs +- Rotation effect (East/West) - Orbit effect (West)
        net_long_change = (self.Wrot*self.timeval_rot) - self.tru_anom + self.spin_history
        new_longs = self.longs[np.newaxis,:] + net_long_change[:,np.newaxis]
        longs_evolve = new_longs % (2.0*pi)
        net_zero_long = net_long_change % (2.0*pi)  # For rotating maps in Orth_Mapper
        return longs_evolve,net_zero_long
    
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
    
    
    def _initial_temperatures(self):
        """Something something else."""
        the_low_case = (0.5*(np.cos(self.longs) + np.absolute(np.cos(self.longs)))*np.sin(self.colat))**0.25
        the_high_case = (np.sin(self.colat)/pi)**0.25
        if np.isnan(self.recirc_effic):
            infinite_eps = self.adv_freq_peri*self.radiate_time
        else:
            infinite_eps = self.recirc_effic
        unity_eps = RecircEfficiency_Convert(infinite_eps)  # Get 0-1 epsilon
        Tvals = unity_eps*the_high_case + (1.0-unity_eps)*the_low_case  # E.B. model parameterization
        Tvals[Tvals<0.01] = 0.01
        return Tvals
    
    def _bitwise_powtwo(self,n):
        """Something something else."""
        return ((n & (n-1)) == 0) and (n > 1)
    
    def _change_T_grid(self,want_temps,old_colat,old_longs,old_NSIDE):
        """Something something else."""
        if self._bitwise_powtwo(old_NSIDE) and self._bitwise_powtwo(self.NSIDE):
            new_temps = hp.pixelfunc.ud_grade(want_temps,self.NSIDE)
        else:
            # Double longs grid to bridge 2*pi-to-0
            doub_colat = np.concatenate((old_colat,old_colat))
            doub_longs = np.concatenate((old_longs,old_longs+2.0*pi))
            old_points = (doub_colat,doub_longs)
            doub_wt = np.concatenate((want_temps,want_temps))
            new_points = (self.colat,self.longs)
            new_temps = interpolate.griddata(old_points,doub_wt,new_points,method='nearest')
        return new_temps
    
    
    def _downpipe_assume_same(self,n):
        """Blah blah blah."""
        return ['_no']*n
    
    def _check_multi_updater(self,things):
        """Blah blah blah."""
        return any(x != '_no' for x in things)
    
    def _using_neworold_param(self,new,old):
        """Blah blah blah."""
        return (new,new) if self._check_single_updater(new) else (old,old)
    
    def _has_param_changed(self,old,new):
        """Blah blah blah."""
        return True if old != new else '_no'
    
    
    def _parameter_pipeline(self,Teff,Rstar,Mstar,
                            Rplanet,smaxis,eccen,arg_peri,bondA,
                            motions,calc_orb,orbval,rotval,
                            radiate_time,recirc_effic,
                            numOrbs,stepsPerOrb,NSIDE,_makenew):
        """Lots of stuff and things."""
        upd_mot,upd_cal,upd_obv,upd_rtv,upd_Po,upd_Pr,upd_Wadv,upd_tv,upd_kE = self._downpipe_assume_same(9)
        
        # Handles '_no' input
        self.Rstar,self.Mstar,self.Rplanet,self.smaxis = self._setup_scaled_quants(Rstar,Mstar,Rplanet,smaxis)
        
        if self._check_single_updater(eccen):
            self.eccen = eccen  # eccentricity
            self._ecc_factor = self._calc_efactor(eccen)  # For scaling ang. vel. at periastron (^-1 for period)
        
        if self._check_single_updater(bondA):
            self.bondA = bondA  # planet Bond albedo
        if self._check_single_updater(Teff):
            self.Teff = Teff  # star effective temp
        if self._check_multi_updater([Teff,Rstar,smaxis,eccen,bondA]):
            self.Tirrad = self._calc_T_irradiation()
        
        ### We add 180 deg to the input "arg_peri", see this method. DON'T GET CONFUSED! :-)
        if self._check_single_updater(arg_peri):
            self.arg_peri = self._modify_arg_peri(arg_peri)
        
        if self._check_multi_updater([motions,calc_orb,orbval,rotval]):
            mots_loc,calo_loc,orbv_loc,rotv_loc = self._parse_motion(motions,calc_orb,orbval,rotval)
            upd_mot = self._has_param_changed(self._last_motions,mots_loc)
            upd_cal = self._has_param_changed(self._last_calc_orb,calo_loc)
            upd_obv = self._has_param_changed(self._last_orbval,orbv_loc)
            upd_rtv = self._has_param_changed(self._last_rotval,rotv_loc)
            
            (self._last_motions,self._last_calc_orb,
             self._last_orbval,self._last_rotval) = mots_loc,calo_loc,orbv_loc,rotv_loc
        else:
            mots_loc,calo_loc,orbv_loc,rotv_loc = (self._last_motions,self._last_calc_orb,
                                                   self._last_orbval,self._last_rotval)
        
        ### Orbital Stuff
        calc_check = calo_loc and self._check_multi_updater([Mstar,smaxis])
        if self._check_multi_updater([upd_mot,upd_cal,upd_obv]) or calc_check:
            old_Porb = '_null' if _makenew else self.Porb
            self.Porb = self._setup_orb_motion(mots_loc,calo_loc,orbv_loc)
            upd_Po = self._has_param_changed(old_Porb,self.Porb)
        else:
            old_Porb = self.Porb

        if self._check_single_updater(stepsPerOrb):
            self.stepsPerOrb = stepsPerOrb
        if self._check_single_updater(numOrbs):
            self.numOrbs = numOrbs
        if self._check_multi_updater([stepsPerOrb,numOrbs,upd_Po]):
            self.timeval,self.trackorbs = self._setup_time_array(_makenew)
            upd_tv,self._should_add_orbtime = True,False

        if self._check_multi_updater([smaxis,eccen,arg_peri,upd_Po]):
            self.kep_E,upd_kE = self._setup_the_orbit(),True
            if not _makenew:
                self.timeval,self.trackorbs = self._match_phase_new_orbit()
                upd_tv,self._should_add_orbtime = True,False

        if self._check_multi_updater([upd_tv,upd_kE,upd_Po]):
            (self.radius,self.orb_pos,self.tru_anom,
             self.Worb,self.alpha,self.frac_litup) = self._calc_orbit_props()
        if self._check_single_updater(upd_kE):
            (self.ecl_time,self.ecl_pos,self.ecl_tru_anom,
             self.trans_time,self.trans_pos,self.trans_tru_anom) = self._find_conjunctions_nodes('c')
            (self.ascend_time,self.ascend_pos,self.ascend_tru_anom,
             self.descend_time,self.descend_pos,self.descend_tru_anom) = self._find_conjunctions_nodes('n')
        if self._check_single_updater(upd_Po):
            self.periast_time,self.apast_time = 0,0.5*self.Porb

        ### Rotational Stuff
        ecc_check = (mots_loc[-1] == 'R') and self._check_single_updater(eccen)
        if self._check_multi_updater([upd_tv,upd_mot,upd_rtv]) or ecc_check:
            self.spin_history,self.timeval_rot = self._reset_rot_times(_makenew)
            self._should_add_rottime = False
        
            old_Prot,old_Wadvec = ('_null','_null') if _makenew else (self.Prot,self.Wadvec)
            self.Prot,self.Wrot,self.Wadvec = self._setup_rot_motion(motions,rotval)
            upd_Pr = self._has_param_changed(old_Prot,self.Prot)
            upd_Wadv = self._has_param_changed(old_Wadvec,self.Wadvec)

        if self._check_multi_updater([eccen,radiate_time,recirc_effic,upd_Wadv]):
            radt_loc,self._last_radiate_time = self._using_neworold_param(radiate_time,self._last_radiate_time)
            rece_loc,self._last_recirc_effic = self._using_neworold_param(recirc_effic,self._last_recirc_effic)
            self.radiate_time,self.recirc_effic = self._setup_radiate_recirc(radt_loc,rece_loc)
        
        ### Atmosphere coordinates
        if self._check_single_updater(NSIDE):
            if not _makenew:
                # For changing T resolution below
                old_colat,old_longs,old_NSIDE = self.colat,self.longs,self.NSIDE
            (self.colat,self.longs,self.pixel_sq_rad,
             self._on_equator,self.NSIDE) = self._setup_colatlong(NSIDE)
        if self._check_multi_updater([NSIDE,upd_Pr,upd_tv,upd_kE]):
            self.longs_evolve,self._net_zero_long = self._calc_longs()
            (self.illumination,self.visibility,
             self.SSP_long,self.SOP_long) = self._calc_vis_illum()

        ### Temperatures
        if _makenew:
            self.Tvals_evolve = np.array([self._initial_temperatures()])
        elif self._check_multi_updater([NSIDE,upd_Po,upd_Pr,upd_Wadv,upd_tv,upd_kE]):
            t_i = -1 if self._has_T_evolved else 0
            want_temps = self.Tvals_evolve[t_i,:]
            if self._check_single_updater(NSIDE):
                want_temps = self._change_T_grid(want_temps,old_colat,old_longs,old_NSIDE)
            self.Tvals_evolve = np.array([want_temps])

        ### Evolve key
        if self._check_multi_updater([radiate_time,recirc_effic,
                                      upd_Po,upd_Pr,upd_Wadv,upd_tv,upd_kE]):
            self._has_T_evolved = False

        return

        
        #########
        
#        mcor_check = self._check_multi_updater([motions,calc_orb,orbval,rotval])
#        if mcor_check or self._check_multi_updater([Mstar,smaxis,eccen]):
#            (mots_loc,calo_loc,orbv_loc,rotv_loc,
#             self._last_motions,self._last_calc_orb,
#             self._last_orbval,self._last_rotval) = self._parse_motion(mcor_check,motions,calc_orb,orbval,rotval)
#            if _makenew:
#                old_Porb,old_Prot,old_adv_freq_peri = '_null','_null','_null'
#            else:
#                old_Porb,old_Prot,old_adv_freq_peri = self.Porb,self.Prot,self.adv_freq_peri
#            self.Porb,self.Prot,self.Wrot,self.adv_freq_peri = self._setup_motion(mots_loc,calo_loc,orbv_loc,rotv_loc)
#            upd_Po = self._has_param_changed(old_Porb,self.Porb)
#            upd_Pr = self._has_param_changed(old_Prot,self.Prot)
#            upd_afp = self._has_param_changed(old_adv_freq_peri,self.adv_freq_peri)
#
#        if self._check_multi_updater([eccen,radiate_time,recirc_effic,upd_afp]):
#            radt_loc,self._last_radiate_time = self._using_neworold_param(radiate_time,self._last_radiate_time)
#            rece_loc,self._last_recirc_effic = self._using_neworold_param(recirc_effic,self._last_recirc_effic)
#            self.radiate_time,self.recirc_effic = self._setup_radiate_recirc(radt_loc,rece_loc)
#
#        ### Time
#        if self._check_single_updater(stepsPerOrb):
#            self.stepsPerOrb = stepsPerOrb
#        if self._check_single_updater(numOrbs):
#            self.numOrbs = numOrbs
#        if self._check_multi_updater([stepsPerOrb,numOrbs,upd_Po]):
#            self.timeval,upd_tv = self._setup_time_array(_makenew),True
#            self._should_add_orbtime = False
#        if self._check_multi_updater([stepsPerOrb,numOrbs,upd_Po,upd_Pr]):
#            self.spin_history,self.timeval_rot = self._reset_rot_times(_makenew)
#            self._should_add_rottime = False
#
#        ### Orbital Stuff
#        if self._check_multi_updater([smaxis,eccen,arg_peri,upd_Po]):
#            self.kep_E,upd_kE = self._setup_the_orbit(),True
#            if not _makenew:
#                self.timeval,upd_tv = self._match_phase_new_orbit(),True
#
#        if self._check_multi_updater([upd_tv,upd_kE]):
#            (self.radius,self.orb_pos,
#             self.tru_anom,self.alpha,self.frac_litup) = self._calc_orbit_props()
#        if self._check_single_updater(upd_kE):
#            (self.ecl_time,self.ecl_pos,self.ecl_tru_anom,
#             self.trans_time,self.trans_pos,self.trans_tru_anom) = self._find_conjunctions_nodes('c')
#            (self.ascend_time,self.ascend_pos,self.ascend_tru_anom,
#             self.descend_time,self.descend_pos,self.descend_tru_anom) = self._find_conjunctions_nodes('n')
#        if self._check_single_updater(upd_Po):
#            self.periast_time,self.apast_time = 0,0.5*self.Porb
#
#        ### Atmosphere coordinates
#        if self._check_single_updater(NSIDE):
#            if not _makenew:
#                # For changing T resolution below
#                old_colat,old_longs,old_NSIDE = self.colat,self.longs,self.NSIDE
#            (self.colat,self.longs,self.pixel_sq_rad,
#             self._on_equator,self.NSIDE) = self._setup_colatlong(NSIDE)
#        if self._check_multi_updater([NSIDE,upd_Pr,upd_tv,upd_kE]):
#            self.longs_evolve,self._net_zero_long = self._calc_longs()
#            (self.illumination,self.visibility,
#             self.SSP_long,self.SOP_long) = self._calc_vis_illum()
#
#        ### Temperatures
#        if _makenew:
#            self.Tvals_evolve = np.array([self._initial_temperatures()])
#        elif self._check_multi_updater([NSIDE,upd_Po,upd_Pr,upd_afp,upd_tv,upd_kE]):
#            t_i = -1 if self._has_T_evolved else 0
#            want_temps = self.Tvals_evolve[t_i,:]
#            if self._check_single_updater(NSIDE):
#                want_temps = self._change_T_grid(want_temps,old_colat,old_longs,old_NSIDE)
#            self.Tvals_evolve = np.array([want_temps])
#
#        ### Evolve key
#        if self._check_multi_updater([radiate_time,recirc_effic,
#                                      upd_Po,upd_Pr,upd_afp,upd_tv,upd_kE]):
#            self._has_T_evolved = False
    
    def _setup_lasts(self):
        """Blah blah blah."""
        self._last_radiate_time = '_null'
        self._last_recirc_effic = '_null'
        self._last_motions = 'perR'
        self._last_calc_orb = True
        self._last_orbval = 1.0
        self._last_rotval = 1.0
        return


    def __init__(self,name='Hot Jupiter',Teff=5778,Rstar=1.0,Mstar=1.0,
                 Rplanet=1.0,smaxis=0.1,eccen=0,arg_peri=0,bondA=0,
                 motions='perR',calc_orb=True,orbval=1.0,rotval=1.0,
                 radiate_time=12.0,recirc_effic=None,
                 numOrbs=3,stepsPerOrb=3600,NSIDE=8):
        
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
        print('Constructing model ... ',end='')
        
        self.name = name
        
        self._has_T_evolved = False
        self._should_add_orbtime = False
        self._should_add_rottime = False
        self._setup_lasts()
        
        self._parameter_pipeline(Teff,Rstar,Mstar,
                                 Rplanet,smaxis,eccen,arg_peri,bondA,
                                 motions,calc_orb,orbval,rotval,
                                 radiate_time,recirc_effic,
                                 numOrbs,stepsPerOrb,NSIDE,_makenew=True)
        
        print('Finished building {:}'.format(self.name))
        return
        
    
    ### PICK UP HERE NEXT TIME, NEED TO ACCOUNT FOR POSSIBLE ARRAY PARAMS ###
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


    ### Changing the system
    def SmartModify_Params(self,name='_no',Teff='_no',Rstar='_no',Mstar='_no',
                           Rplanet='_no',smaxis='_no',eccen='_no',arg_peri='_no',bondA='_no',
                           motions='_no',calc_orb='_no',orbval='_no',rotval='_no',
                           radiate_time='_no',recirc_effic='_no',
                           numOrbs='_no',stepsPerOrb='_no',NSIDE='_no'):
        """Change your stuff around!"""
        print('Starting smart mods ... ',end='')
        if self._check_single_updater(name):
            self.name = name
        
        self._parameter_pipeline(Teff,Rstar,Mstar,
                                 Rplanet,smaxis,eccen,arg_peri,bondA,
                                 motions,calc_orb,orbval,rotval,
                                 radiate_time,recirc_effic,
                                 numOrbs,stepsPerOrb,NSIDE,_makenew=False)
        
        print('Finished modifying {:}'.format(self.name))
        return
    
    
    ### Draw orbit
    
    def _orbit_auscale(self,pos):
        return pos/self.astro_unit
    
    def _orbit_scatter(self,axorb,pos,ol_sty):
        au_pos = self._orbit_auscale(pos)
        color,mark,ize,lab = ol_sty
        axorb.scatter(au_pos[0],au_pos[2],c=color,marker=mark,s=ize,edgecolors='k',zorder=2,label=lab)
        return
    
    def _orbit_window(self,axorb,au_pos):
        au_sep = max(np.ptp(au_pos[:,0]),np.ptp(au_pos[:,2]))
        au_cent = self.kep_E.xyzCenter()/self.astro_unit
        
        f = 1.1  # Padding factor
        dist = f*(0.5*au_sep)  # Distance from center of ellipse
        axorb.set_xlim(au_cent[0]-dist,au_cent[0]+dist)
        axorb.set_ylim(au_cent[2]-dist,au_cent[2]+dist)
        
        axorb.set_title('Orbit of '+self.name)
        axorb.set_xlabel('Distance from star (AU)')
        axorb.set_ylabel('Distance from star (AU)')
        return
    
    def _orbit_lines(self,axorb,pos):
        au_pos = self._orbit_auscale(pos)
        axorb.plot([0,au_pos[0]],[0,au_pos[2]],c='0.5',ls=':',zorder=0)
        return
    
    def Draw_OrbitOverhead(self,show_legend=True,_combo=False,_axuse=None,_phxyz=None):
        """Something something else."""
        if _combo:
            axorb = _axuse
        else:
            fig_orbit,axorb = plt.subplots(figsize=(7,7))
        
        i_one = int(self.stepsPerOrb+1)
        au_pos = self.orb_pos[:i_one]/self.astro_unit
        axorb.plot(au_pos[:,0],au_pos[:,2],c='k',zorder=1)  # Overhead is x-z plane
        
        # Star has marker and line styles
        color,m_ls,ize = orbloc_styles_['star'][:-1]
        mark,ls = m_ls
        axorb.scatter(0,0,c=color,marker=mark,s=4*ize,edgecolors='k',linestyle=ls,zorder=1)
        
        self._orbit_scatter(axorb,self.trans_pos,orbloc_styles_['transit'])
        self._orbit_lines(axorb,self.trans_pos)
        self._orbit_scatter(axorb,self.ecl_pos,orbloc_styles_['eclipse'])
        self._orbit_lines(axorb,self.ecl_pos)
        
        # Need self.kep_E.xyzNodes_LOSZ() for anything??
        self._orbit_scatter(axorb,self.ascend_pos,orbloc_styles_['ascend'])
        self._orbit_lines(axorb,self.ascend_pos)
        self._orbit_scatter(axorb,self.descend_pos,orbloc_styles_['descend'])
        self._orbit_lines(axorb,self.descend_pos)
        
        periastron,apastron = self.kep_E.xyzPeriastron(),self.kep_E.xyzApastron()
        self._orbit_scatter(axorb,periastron,orbloc_styles_['periast'])
        self._orbit_scatter(axorb,apastron,orbloc_styles_['apast'])
        
        if _combo and isinstance(_phxyz,np.ndarray):
            self._orbit_scatter(axorb,_phxyz,orbloc_styles_['phase'])
        
        self._orbit_window(axorb,au_pos)
        if show_legend:
            axorb.legend(loc='best')

        axorb.set_aspect('equal')
        
        if not _combo:
            fig_orbit.tight_layout()
            self.fig_orbit = fig_orbit
            plt.show()
        return
    
    
    ### Differential Equation

    def _update_params_before_evolve(self):
        """Something something else."""
        # Orbital
        if self._should_add_orbtime:
            self.timeval += self.Porb*self.numOrbs
        
            (self.radius,self.orb_pos,
             self.tru_anom,self.alpha,self.frac_litup) = self._calc_orbit_props()

            self.longs_evolve,self._net_zero_long = self._calc_longs()
             
            (self.illumination,self.visibility,
             self.SSP_long,self.SOP_long) = self._calc_vis_illum()
        
        o_start,o_end = self.timeval[[0,-1]]/self.Porb
        
        # Rotational
        if self._should_add_rottime:
            self.timeval_rot += self.Porb*self.numOrbs

        return o_start,o_end
    
    
    ### PICK UP HERE TOO, THIS METHOD WILL NEED EDITS NOW THAT Wadvec EXISTS ###
    def _diff_eq_tempvals(self,start_Tvals):
        """Something something else."""
        Tvals_evolve = np.zeros(self.longs_evolve.shape)
        Tvals_evolve[0,:] += start_Tvals
        
        if self.eccen == 0:
            if (abs(self.recirc_effic) <= 10**(-4)):
                Tvals_evolve = ((1.0-self.bondA)*self.illumination)**(0.25)
            else:
                # Here advective frequency is constant- sign spcifies direction atmosphere rotates.
                the_sign = -1.0 if self.adv_freq_peri < 0 else 1.0
                delta_longs = (self.longs_evolve[1:,:] - self.longs_evolve[:-1,:]) % (the_sign*2.0*pi)
                
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
    
    ## ADD WAY TO AUTOMATE EVOLVING UNTIL SOME EQUILIBRIUM IS REACHED? ##
    def Evolve_AtmoTemps(self):
        """Something something else."""
        t_i,s = (-1,'Re-heating') if self._has_T_evolved else (0,'Heating')
        start_Tvals = self.Tvals_evolve[t_i,:]
        
        o_start,o_end = self._update_params_before_evolve()
        print(s+' {:}, orbits {:.2f} to {:.2f} ... '.format(self.name,o_start,o_end),end='')

        self.Tvals_evolve = self._diff_eq_tempvals(start_Tvals)
        self._has_T_evolved = True
        self._should_add_orbtime = True
        self._should_add_rottime = True
        
        print('Evolving complete')
        return
    
    
    ### Temperature Map
    
    def _final_orbit_index(self):
        """Something something else."""
        # +1 so initial phase is not included twice.
        return int(round(self.stepsPerOrb*(self.numOrbs-1))) + 1
    
    def _orth_bounds(self,force_contrast,heat_map):
        """Something something else."""
        if force_contrast:
            # Just a little larger than the temperature bounds
            low = 0.001*np.floor(1000*np.amin(heat_map))
            high = 0.001*np.ceil(1000*np.amax(heat_map))
        else:
            low,high = 0,1.0
        return low,high
    
    def _orth_grat_clats(self,axmap,xcen,xfac,ylev,mc,sc):
        """Something something else."""
        i_eq = int((xfac.size - 1)/2)
        for xc in xcen:
            for i in np.arange(xfac.size):
                l_c,l_s = (mc,'-') if i == i_eq else (sc,':')
                axmap.plot([xc-xfac[i],xc+xfac[i]],[ylev[i],ylev[i]],c=l_c,ls=l_s,lw=1,zorder=3)
        return
    
    def _orth_gl_plot(self,axmap,xcen,ra,sincla,coscla,l_c,l_s,star,ol_sty):
        """Something something else."""
        xlon = xcen + sincla*np.sin(np.radians(ra))
        axmap.plot(xlon,coscla,c=l_c,ls=l_s,lw=1,zorder=3)
        if star:
            color,m_ls,ize = ol_sty[:-1]
            mark,ls = m_ls
            axmap.scatter(xcen+np.sin(np.radians(ra)),0,c=color,marker=mark,s=ize,
                          edgecolors='k',linestyle=ls,zorder=4)
        return
    
    def _orth_grat_longs(self,axmap,far_side,rel_ssp,rel_angs,sincla,coscla,mc,sc):
        """Something something else."""
        xcen = -1 if far_side else 0
        for ra in rel_angs:
            if ra == rel_ssp:
                l_c,l_s,star = mc,'-',True
            elif ra == rel_ssp + 180:
                l_c,l_s,star = mc,'-',False
            else:
                l_c,l_s,star = sc,':',False
            
            if np.cos(np.radians(ra)) >= 0:
                self._orth_gl_plot(axmap,xcen,ra,sincla,coscla,l_c,l_s,star,orbloc_styles_['star'])
            if far_side and (np.cos(np.radians(ra)) <= 0):
                self._orth_gl_plot(axmap,1,-1*ra,sincla,coscla,l_c,l_s,star,orbloc_styles_['star'])
        return

    def _orth_graticule(self,axmap,zero_to_sop,far_side):
        """Something something else."""
        mc,sc = '0.75','0.1'  # Main/secondary colors
        
        xcen = [-1,1] if far_side else [0]
        self._orth_grat_clats(axmap,xcen,grat_xfactor_,grat_ylevels_,mc,sc)
        
        rel_ssp = -zero_to_sop
        rel_angs = np.linspace(rel_ssp,rel_ssp + 330,12)
        self._orth_grat_longs(axmap,far_side,rel_ssp,rel_angs,grat_sinclats_,grat_cosclats_,mc,sc)
        return

    def _orth_cbar(self,axmap,new_map,low,high,far_side,_combo,_cax):
        """Something something else."""
        if _combo:
            cb = plt.colorbar(new_map,cax=_cax,orientation='vertical',ticks=[low,high])
            tx,ty,r = 1.75,0.5,'vertical'
        else:
            wid = lambda fs: 0.4 if fs else 0.6
            cb = plt.colorbar(new_map,ax=axmap,orientation='horizontal',shrink=wid(far_side),
                              aspect=25,ticks=[low,high],pad=0.05,fraction=0.1)
            tx,ty,r = 0.5,-1.0,'horizontal'
        
        unit_of_T = r'Normalized Temperature $(T \ / \ T_{0})$'
        cb.ax.text(tx,ty,unit_of_T,rotation=r,fontsize='large',ha='center',va='center',
                   transform=cb.ax.transAxes)
        return
    
    def Orth_Mapper(self,phase,relative_periast=False,force_contrast=False,far_side=False,
                    _combo=False,_axuse=None,_cax=None,_i_phase=None):
        """Something something else."""
        if _i_phase == None:
            fin_orb_start = self._final_orbit_index()
            # Find closest position to phase, given start position.
            if relative_periast:
                diff_phase = np.radians(phase) - self.tru_anom[fin_orb_start:]
            else:
                diff_phase = np.radians(phase - self.alpha[fin_orb_start:])
            i_want = np.argmax(np.cos(diff_phase)) + fin_orb_start
        else:
            i_want = _i_phase
        
        # Align map for orthview because pixel (i.e. atmo) longitudes are not static.
        # Undo implicit drift of the SSP (longitude 0), then rotate to the SOP.
        # It's very easy to get confused by this!!
        zero_long = np.degrees(self._net_zero_long[i_want])
        zero_to_sop = np.degrees(self.SOP_long[i_want])
        sop_rot = (-zero_long + zero_to_sop) % 360
        
        heat_map = self.Tvals_evolve[i_want,:]
        low,high = self._orth_bounds(force_contrast,heat_map)
        
        # Get the picture from orthview to re-style; lucky 13!!
        xpix,hsiz,xval = (2400,14,2) if far_side else (1200,7,1)
        pic_map = hp.visufunc.orthview(fig=13,map=heat_map,rot=(sop_rot,0,0),flip='geo',
                                       min=low,max=high,cmap=inferno_mod_,
                                       half_sky=not(far_side),xsize=xpix,return_projected_map=True)
        plt.close(13)
        
        if _combo:
            axmap = _axuse
        else:
            fig_orth,axmap = plt.subplots(figsize=(hsiz,7))

        new_map = axmap.imshow(pic_map,origin='lower',extent=[-xval,xval,-1,1],
                               vmin=low,vmax=high,cmap=inferno_mod_)
        self._orth_graticule(axmap,zero_to_sop,far_side)

        ### Have my custom graticule now, but keeping this for posterity:
        # Seems like graticule + orthview can throw out two invalid value warnings.
        # Both pop up when *half_sky* is True, only one when it's False.
        # Then again, if the numbers in *rot* have > 1 decimal place,
        # those warnings can disappear! It's something weird in healpy's
        # projector.py and projaxes.py. I'm suppressing both warnings for now.
#        with warnings.catch_warnings():
#            warnings.filterwarnings('ignore',message='invalid value encountered in greater')
#            hp.visufunc.graticule(local=True,verbose=True)

        if relative_periast:
            descrip = r' $%.2f^{\circ}$ from periastron' % (np.degrees(self.tru_anom[i_want]) % 360)
        else:
            descrip = r' at $%.2f^{\circ}$ orbital phase' % self.alpha[i_want]
        axmap.set_title(self.name + descrip)
        self._orth_cbar(axmap,new_map,low,high,far_side,_combo,_cax)

        if far_side:
            axmap.text(-1,-1.06,'Observer Side',size='large',ha='right',va='center')
            axmap.text(1,-1.06,'Far Side',size='large',ha='left',va='center')

        axmap.axes.get_xaxis().set_visible(False)
        axmap.axes.get_yaxis().set_visible(False)
        axmap.axis('off')

        if not _combo:
            fig_orth.tight_layout()
            self.fig_orth = fig_orth
            plt.show()
        elif _i_phase == None:
            return self.orb_pos[i_want,:]
        
        return


    def _combo_faxmaker(self,sr,sc):
        """Blah blah blah."""
        Nc,d = (2*sc)+1,sc/sr
        f_com = plt.figure(figsize=(Nc/d,sr))
        _axl = plt.subplot2grid((1,Nc),(0,0),rowspan=1,colspan=sc,fig=f_com)
        _axr = plt.subplot2grid((1,Nc),(0,sc),rowspan=1,colspan=sc,fig=f_com)
        _cax = plt.subplot2grid((1,Nc),(0,2*sc),rowspan=1,colspan=1,fig=f_com)
        return f_com,_axl,_axr,_cax

    def Combo_OrbitOrth(self,phase,relative_periast=False,show_legend=True,force_contrast=False):
        """Blah blah blah."""
        fig_orborth,_axorb,_axmap,_cax = self._combo_faxmaker(sr=7,sc=14)
        
        # Return phase position before drawing orbit
        _phxyz = self.Orth_Mapper(phase,relative_periast,force_contrast,far_side=False,
                                  _combo=True,_axuse=_axmap,_cax=_cax)
        
        self.Draw_OrbitOverhead(show_legend,_combo=True,_axuse=_axorb,_phxyz=_phxyz)
            
        fig_orborth.tight_layout(w_pad=1)
        self.fig_orborth = fig_orborth
        plt.show()
        return
    

    ### Blackbody methods
    
    def _waveband_to_lowup(self,wave_microns,band_microns):
        """Blah blah blah."""
        return wave_microns-(0.5*band_microns),wave_microns+(0.5*band_microns)
    
    def _um_to_m(self,microns):
        """Blah blah blah."""
        return microns*(10**(-6))

    
    def _blackbody_bolometric(self,temperature):
        """Units of W/m^2"""
        return self.stef_boltz*(temperature**4)


    def _plancks_law(self,wavelength,temperature):
        """Blah blah blah."""
        try:
            xpon = self.planck*self.speed_light/(wavelength*self.boltz*temperature)
        except ZeroDivisionError:
            xpon = np.inf
        # Leading pi from integral over solid angle, so wave/bolo units match.
        return pi*(2.0*self.planck*(self.speed_light**2)/(wavelength**5))*(1.0/np.expm1(xpon))
    
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
        lower_i = np.argmin(np.absolute(preplancked_wavelengths_ - lower_microns))
        upper_i = np.argmin(np.absolute(preplancked_wavelengths_ - upper_microns))
        if lower_i == upper_i:
            if upper_i == (preplancked_waveN_ - 1):
                lower_i -= 1
            else:
                upper_i += 1
        return lower_i,upper_i

    def _blackbody_wavematchplancked(self,lower_microns,upper_microns,temperature):
        """_blackbody_wavelength but using preplancked wavelengths."""
        lower_i,upper_i = self._get_wave_indices(lower_microns,upper_microns)
        low_mic = preplancked_wavelengths_[lower_i]
        upp_mic = preplancked_wavelengths_[upper_i]
        bb_values,bb_errors = self._blackbody_wavelength(low_mic,upp_mic,temperature)
        return bb_values,bb_errors

    
    def _change_temper_vals_to_inds(self,temper_val):
        """Cool, looks like this works well!"""
        return preplancked_temper_valueinds_[temper_val.astype(int)]
    
    def _get_temper_indices(self,temperature):
        """Blah blah blah."""
        high_t = preplancked_temperatures_[-1]
        
        temper_val = np.around(np.atleast_1d(temperature))
        # With "_blackbody_wavematchplancked", may not need these checks.
        high_check = (temper_val > high_t)
        temper_val[high_check] = high_t
        zero_check = (temper_val < 0)
        temper_val[zero_check] = 0
        
        temper_i = self._change_temper_vals_to_inds(temper_val)
        return temper_i

    def _blackbody_preplancked(self,lower_microns,upper_microns,temperature):
        """Units of W/m^2"""
        lower_i,upper_i = self._get_wave_indices(lower_microns,upper_microns)
        temper_i = self._get_temper_indices(temperature)

        chosen_integrals = np.sum(preplancked_integrals_[:,lower_i:upper_i],axis=1)
        return chosen_integrals[temper_i]
    

    def Observed_Flux(self,wave_band=False,a_microns=6.5,b_microns=9.5,
                      kind='obs',run_integrals=False,bolo=False,separate=False):
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
            star_bb = self._blackbody_bolometric(self.Teff)
            planet_bb = self._blackbody_bolometric(planet_Treal)
        elif run_integrals:
            star_bb,_foo = self._blackbody_wavelength(lower_microns,upper_microns,self.Teff)
            planet_bb,_foo = self._blackbody_wavelength(lower_microns,upper_microns,planet_Treal)
        else:
            high_t = preplancked_temperatures_[-1]
            if self.Teff > high_t:
                print('Observed_Flux: star T_eff is higher than PrePlancked values- integrating instead.')
                star_bb,_foo = self._blackbody_wavematchplancked(lower_microns,upper_microns,self.Teff)
            else:
                star_bb = self._blackbody_preplancked(lower_microns,upper_microns,self.Teff)
            
            high_check = (planet_Treal > high_t)
            if np.any(high_check):
                message = 'Observed_Flux: {:.5f} of planet T\'s are higher than PrePlancked values- integrating these.'
                print(message.format(np.sum(high_check)/high_check.size))
                planet_bb = np.zeros(planet_Treal.shape)
                planet_bb[high_check],_foo = self._blackbody_wavematchplancked(lower_microns,upper_microns,planet_Treal[high_check])
                good_check = np.logical_not(high_check)
                planet_bb[good_check] = self._blackbody_preplancked(lower_microns,upper_microns,planet_Treal[good_check])
            else:
                planet_bb = self._blackbody_preplancked(lower_microns,upper_microns,planet_Treal)

        star_flux = (star_vis*star_bb)*(self.Rstar**2)
        planet_flux = (np.sum(planet_vis*planet_bb,axis=1)*self.pixel_sq_rad)*(self.Rplanet**2)
        
        if separate:
            return planet_flux,star_flux
        else:
            return planet_flux/star_flux


    ### Draw Light Curve
    
    def _light_indices(self,begins):
        """Blah blah blah."""
        # _final_orbit_index has +1 so initial phase is not included twice.
        fin_orb_start = self._final_orbit_index()
        
        if begins == 'periast':
            fi_end = np.argmax(np.cos(self.tru_anom[fin_orb_start:]))
        elif begins == 'apast':
            fi_end = np.argmin(np.cos(self.tru_anom[fin_orb_start:]))
        elif begins == 'transit':
            fi_end = np.argmax(np.cos(np.radians(self.alpha[fin_orb_start:])))  # At 0 phase
        elif begins == 'eclipse':
            fi_end = np.argmin(np.cos(np.radians(self.alpha[fin_orb_start:])))  # At 180 phase
        elif begins == 'ascend':
            fi_end = np.argmax(np.sin(np.radians(self.alpha[fin_orb_start:])))  # At 90 phase
        elif begins == 'descend':
            fi_end = np.argmin(np.sin(np.radians(self.alpha[fin_orb_start:])))  # At 270 phase

        i_end = fi_end + fin_orb_start
            
        i_start = i_end - int(self.stepsPerOrb)
        i_end += 1  # To have initial phase repeated
        return i_start,i_end
    
    def _relative_time(self,want_t,t_start):
        """Blah blah blah."""
        return (want_t - t_start)/self.Porb
    
    def _light_times(self,i_start,i_end):
        """Blah blah blah."""
        t_act = self.timeval[i_start:i_end]
        t_start,t_end = t_act[0],t_act[-1]
        o_start = np.floor(t_start/self.Porb)  # Orbit light curve starts (0 based)
        
        t_rel = self._relative_time(t_act,t_start)
        return t_act,t_start,t_end,o_start,t_rel
    
    def _prop_plotter(self,axlig,t_a,t_start,f_terp,ol_sty,y_mark,_combo,_inc):
        """Blah blah blah."""
        f_v = f_terp(t_a)
        t_r = self._relative_time(t_a,t_start)
        
        color = ol_sty[0]
        axlig.plot([t_r,t_r],[0,f_v],c=color,ls='--',zorder=2)
        if _combo:
            mark,ize = ol_sty[1:3]
            lab = ol_sty[-1] if _inc else '_null'
            axlig.scatter(t_r,y_mark,c=color,marker=mark,s=ize,edgecolors='k',zorder=2,label=lab)
        return
    
    def _prop_plotcheck(self,axlig,prop_time,o_start,t_start,t_end,f_terp,
                        ol_sty,y_mark,_combo):
        """Blah blah blah."""
        t_a = prop_time+(o_start*self.Porb)
        _inc = True  # For putting marker in legend
        while t_a <= t_end:
            if t_a >= t_start:
                self._prop_plotter(axlig,t_a,t_start,f_terp,ol_sty,y_mark,_combo,_inc)
                _inc = False  # No double-listed markers!
            t_a += self.Porb
        return
    
    def _light_window(self,axlig,lc_high,f_y,ol_sty):
        """Blah blah blah."""
        axlig.set_ylim(-f_y*lc_high,(1+f_y)*lc_high)
        
        axlig.set_title('Light Curve of '+self.name)
        axlig.set_xlabel('Time from '+ol_sty[-1]+' (orbits)')
        axlig.set_ylabel('Flux ( planet / star )')
        return

    def Draw_LightCurve(self,wave_band=False,a_microns=6.5,b_microns=9.5,
                        run_integrals=False,bolo=False,begins='periast',
                        _combo=False,_axuse=None,_phase=None,_relperi=None):
        """Blah blah blah."""
        if begins not in self._accept_begins:
            print('Draw_LightCurve error: strings for *begins* are')
            print(self._accept_begins)
            plt.close()  # Remove WIP plot if combo method
            return
        
        lightcurve_flux = self.Observed_Flux(wave_band,a_microns,b_microns,
                                             'obs',run_integrals,bolo,False)
        if _combo:
            axlig = _axuse
        else:
            fig_light,axlig = plt.subplots(figsize=(7,7))
        
        i_start,i_end = self._light_indices(begins)
        lcf_use = lightcurve_flux[i_start:i_end]
        t_act,t_start,t_end,o_start,t_rel = self._light_times(i_start,i_end)
        
        # Get correct index/time for _phase in the combo plot.
        if _phase != None:
            # Find closest position to _phase, given light curve.
            if _relperi:
                _diff_phase = np.radians(_phase) - self.tru_anom[i_start:i_end]
            else:
                _diff_phase = np.radians(_phase - self.alpha[i_start:i_end])
            _i_phase = np.argmax(np.cos(_diff_phase)) + i_start
            _time_phase = self.timeval[_i_phase] % self.Porb

        axlig.plot(t_rel,lcf_use,c='k',lw=2,zorder=3)
        axlig.axhline(0,c='0.5',ls=':',zorder=1)
        
        f_terp = interpolate.interp1d(t_act,lcf_use)
        lc_high,f_y = np.amax(lcf_use),0.05  # y-axis scale factor
        y_mark = -0.5*(f_y*lc_high)
        
        self._prop_plotcheck(axlig,self.trans_time,o_start,t_start,t_end,f_terp,
                             orbloc_styles_['transit'],y_mark,_combo)
        self._prop_plotcheck(axlig,self.ecl_time,o_start,t_start,t_end,f_terp,
                             orbloc_styles_['eclipse'],y_mark,_combo)
        if _combo:
            self._prop_plotcheck(axlig,self.ascend_time,o_start,t_start,t_end,f_terp,
                                 orbloc_styles_['ascend'],y_mark,_combo)
            self._prop_plotcheck(axlig,self.descend_time,o_start,t_start,t_end,f_terp,
                                 orbloc_styles_['descend'],y_mark,_combo)
            self._prop_plotcheck(axlig,self.periast_time,o_start,t_start,t_end,f_terp,
                                 orbloc_styles_['periast'],y_mark,_combo)
            self._prop_plotcheck(axlig,self.apast_time,o_start,t_start,t_end,f_terp,
                                 orbloc_styles_['apast'],y_mark,_combo)
            if _phase != None:
                self._prop_plotcheck(axlig,_time_phase,o_start,t_start,t_end,f_terp,
                                     orbloc_styles_['phase'],y_mark,_combo)
        
        self._light_window(axlig,lc_high,f_y,orbloc_styles_[begins])
        
        if not _combo:
            fig_light.tight_layout()
            self.fig_light = fig_light
            plt.show()
        elif _phase != None:
            return _i_phase

        return
    
    
    def Combo_OrbitLC(self,show_legend=True,wave_band=False,a_microns=6.5,b_microns=9.5,
                      run_integrals=False,bolo=False,begins='periast'):
        """Blah blah blah."""
        fig_orblc = plt.figure(figsize=(14,7))
        
        _axorb = plt.subplot(121)
        self.Draw_OrbitOverhead(show_legend,_combo=True,_axuse=_axorb)
        
        _axlig = plt.subplot(122)
        self.Draw_LightCurve(wave_band,a_microns,b_microns,run_integrals,bolo,
                             begins,_combo=True,_axuse=_axlig)
        
        fig_orblc.tight_layout(w_pad=2)
        self.fig_orblc = fig_orblc
        plt.show()
        return
    
    
    def Combo_LCOrth(self,phase,relative_periast=False,force_contrast=False,
                     wave_band=False,a_microns=6.5,b_microns=9.5,
                     run_integrals=False,bolo=False,begins='periast',show_legend=True):
        """Blah blah blah."""
        fig_lcorth,_axlig,_axmap,_cax = self._combo_faxmaker(sr=7,sc=14)
        
        # Return correct phase index to override calc in Orth_Mapper
        _i_phase = self.Draw_LightCurve(wave_band,a_microns,b_microns,run_integrals,bolo,
                                        begins,_combo=True,_axuse=_axlig,
                                        _phase=phase,_relperi=relative_periast)
        
        # Check if light curve quit with bad *begins*
        if _i_phase == None:
            return
        
        self.Orth_Mapper(phase,relative_periast,force_contrast,far_side=False,
                         _combo=True,_axuse=_axmap,_cax=_cax,_i_phase=_i_phase)
                         
        if show_legend:
            _axlig.legend(loc='best')
      
        fig_lcorth.tight_layout(w_pad=1)
        self.fig_lcorth = fig_lcorth
        plt.show()
        return


    ### Temperature methods

    def Calc_MaxEastWest_Temps(self):
        """Something.
            
        Something else.
        
        """
        start_time = int(self.stepsPerOrb*(self.numOrbs-1))
        
        alphas = self.alpha[start_time:]
        Tevo_eq = self.Tvals_evolve[start_time:,self._on_equator]
        long_eq = self.longs_evolve[start_time:,self._on_equator]
        # To pair timestep with values
        i_time = np.arange(Tevo_eq.shape[0])
        
        i_max = np.argmax(Tevo_eq,axis=1)
        Tnorm_max = Tevo_eq[i_time,i_max]
        shifts_max = long_eq[i_time,i_max]
        
        # East at long = pi/2 (prograde dusk); doesn't matter that angles wrap.
        i_east = np.argmin(np.absolute(long_eq - (pi/2)),axis=1)
        Tnorm_east = Tevo_eq[i_time,i_east]
        
        # West at long = 3*pi/2 (prograde dawn); ditto.
        i_west = np.argmin(np.absolute(long_eq - (3*pi/2)),axis=1)
        Tnorm_west = Tevo_eq[i_time,i_west]
        
        return Tnorm_max,Tnorm_east,Tnorm_west,alphas,shifts_max

    
    def _max_temper(self,eps):
        """Something."""
        if eps == 0:
            func = 0
        else:
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
        if eps == 0:
            val = 0
        else:
            y_0 = 0.69073
            y_1 = 7.5534
            val = ((pi**2)*((1.0 + (y_0/eps))**(-8)) + y_1*(eps**(-8/7)))**(-1/8)
        return val
    
    def _dawn_temper(self,eps):
        """Something."""
        if eps == 0:
            val = 0
        else:
            val = (pi + (3.0*pi/eps)**(4/3))**(-1/4)
        return val
    
    
    def Approx_MaxEastWest_Temps(self,epsilon="self"):
        """Blah blah blah
        
        Some bozo.
        
        """
        if isinstance(epsilon,(float,int)):
            eps = epsilon
        else:
            eps = self.recirc_effic
        
        if eps >= 0:
            # Prograde
            Tnorm_max = self._max_temper(eps)
            Tnorm_east = self._dusk_temper(eps)
            Tnorm_west = self._dawn_temper(eps)
        else:
            # Retrograde --> swap dusk/dawn
            Tnorm_max = self._max_temper(abs(eps))
            Tnorm_west = self._dusk_temper(abs(eps))
            Tnorm_east = self._dawn_temper(abs(eps))
        
        return Tnorm_max,Tnorm_east,Tnorm_west
