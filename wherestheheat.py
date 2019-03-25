# -*- coding: utf-8 -*-
"""



**Put your cleaned-up module docstring up here.**



Legacy notes from Diana's version 6

Created a subclass called fitter. It'll take results from parcel class, 
stitch together the input time array with 1 period of the parcel time array and 
calculate values for just that one period. It depends only on time, tau_rad and 
wadv. It's quicker... and doesn't return any maps and crap.

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

from mpl_toolkits.axes_grid1 import make_axes_locatable

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
                  'phase':['0.8','o',150,'Planet Phase'],
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
        d = lambda n: (' '*4)*n
        print('RecircEfficiency_Convert error: strings for *kind* are')
        print(d(1)+'\"infinite\" to convert 0-inf --> 0-1, or')
        print(d(1)+'\"unity\" to convert 0-1 --> 0-inf.')
        return

    return new_epsilon


class parcel(object):

    """This class allows you to create a planet object and assign it appropriate orbital 
    and planetary parameters. It uses class functions to calculate the planetary phase curve
    (emmitted flux) for an arbitrary number of orbits (default is 3). Use this class if you 
    want to make figures.

    Params that you might want to change 
    ------------------------------------
        numOrbs
                Number of orbital periods we will integrate for. Default is 3.
                Might need more for large values of the radiative time scale because you want the DE
                to have time to reach a stable state.
                
        stepsPerOrb
                int; number of time steps. Default is 3600.
            
        NSIDE
                healpix parameter for number of pixels on the planet surface.
                NPIX = 12*(NSIDE**2) (ex: NPIX = 192 --> NSIDE = 4 ; NPIX = 798 --> NSIDE = 8)
                
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
    _accept_rotval = ('time','periS','periC','phaseS','phaseC','logist')
    _accept_begins = ('transit','eclipse','ascend','descend','periast','apast')
    
    _tab_spaces = 4
    _method_delim = '* * * * * * *'
    
    
    def _shift_tab(self,amt=None):
        """Blah blah blah."""
        self._tab_level = 0 if amt == None else max(0,int(amt)+self._tab_level)
        return
    
    def _indent(self):
        return (' '*self._tab_spaces)*self._tab_level

    def _dyn_print(self,words,amt=None,end='\n'):
        """Blah blah blah."""
        if end == '\n':
            print(self._indent()+words)
            self._shift_tab(amt=amt)
        else:
            print(words,end=end)
        return
    
    
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
    
    
    def _orbrot_entering(self,ty_tup,word,blank,ok,last,cant):
        """Something some thing."""
        orbrot = None
        while not isinstance(orbrot,ty_tup):
            i = input(self._indent()+'Enter '+word+blank)
            if i == '':
                self._dyn_print(ok,amt=0)
                orbrot = last
                break
            try:
                orbrot = eval(i)
            except:
                self._dyn_print(cant,amt=0)
        return orbrot
    
    def _parse_motion(self,motions,calc_orb,orbval,rotval):
        """Blah blah blah."""
        mots_loc,calo_loc,orbv_loc,rotv_loc = motions,calc_orb,orbval,rotval
        cur = lambda v: self._dyn_print('Current value is {:}.'.format(v),amt=0)
        blank = '; blank to keep current: '
        ok = 'OK, keeping current value.'
        cant = 'Cannot eval, try again.'
        gap = False
        
        if mots_loc not in self._accept_motions:
            gap = True
            self._dyn_print('Constructor warning: *motions* strings are '+str(self._accept_motions)+'.',amt=1)
            cur(self._last_motions)
            while mots_loc not in self._accept_motions:
                mots_loc = input(self._indent()+'Enter a *motions* (no quotes)'+blank)
                if mots_loc == '':
                    self._dyn_print(ok,amt=0)
                    mots_loc = self._last_motions
                    break
            self._shift_tab(amt=-1)

        if calo_loc not in [True,False]:
            gap = True
            self._dyn_print('Constructor warning: *calc_orb* is boolean.',amt=1)
            cur(self._last_calc_orb)
            boo = None
            while boo not in ['T','F']:
                b = input(self._indent()+'Enter a *calc_orb* [T/F]'+blank).capitalize()
                if b == '':
                    self._dyn_print(ok,amt=0)
                    calo_loc = self._last_calc_orb
                    break
                boo = b[0]
                if boo in ['T','F']:
                    calo_loc = True if boo == 'T' else False
            self._shift_tab(amt=-1)

        # Check if *calc_orb* is False...
        if not calo_loc:
            if not isinstance(orbv_loc,(float,int)):
                gap = True
                self._dyn_print('Constructor warning: *orbval* is a float or integer.',amt=1)
                cur(self._last_orbval)
                orbv_loc = self._orbrot_entering((float,int),'an *orbval*',blank,ok,self._last_orbval,cant)
                self._shift_tab(amt=-1)
        else:
            # ...otherwise *orbval* doesn't matter.
            orbv_loc = orbv_loc if isinstance(orbv_loc,(float,int)) else self._last_orbval
        
        if not isinstance(rotv_loc,(float,int,list)):
            gap = True
            self._dyn_print('Constructor warning: *rotval* is a float, integer, or list (see docs).',amt=1)
            cur(self._last_rotval)
            rotv_loc = self._orbrot_entering((float,int,list),'a *rotval*',blank,ok,self._last_rotval,cant)
            self._shift_tab(amt=-1)

        if gap:
            print('')

        return mots_loc,calo_loc,orbv_loc,rotv_loc

    def _verify_variable_rotval(self,rotval):
        """Blah blah blah."""
        verify_rvL = []  # List to collect each verify

        if len(rotval) > 0:  # Blanks sent straight to wizard!
            ver_const = isinstance(rotval[0],(float,int,str))  # Constant term
            verify_rvL.append(ver_const)
            
            for rv in rotval[1:]:
                try:  # List-like entries might pass...
                    ver_kind = rv[0] in self._accept_rotval
                    
                    nl = len(rv[1:])
                    rvA = np.asarray(rv[1:])
                    sA = rvA.size
                    # Is it numbers and either [c1,c2,c3,...] or [[ord1,c1,off1],...] ?
                    ver_nums = np.issubdtype(rvA.dtype,np.number) and ((sA == nl) or (sA == 3*nl))
                    
                    verify_rvL.append(np.all([ver_kind,ver_nums]))
                
                except:  # ...and everything else will fail. :-)
                    verify_rvL.append(False)
        
        return verify_rvL
    
    def _print_pieces(self,stuff):
        """Blah blah blah."""
        n = 1
        for p in stuff:
            amt = -1 if n == len(stuff) else 0
            self._dyn_print(str(p),amt=amt)
            n += 1
        return
    
    def _ask_question(self,dflt,phrase):
        """Blah blah blah."""
        calls = ' [*y/n]: ' if dflt == 'y' else ' [y/n*]: '
        
        answer = None
        while answer not in ['y','n']:
            i = input(self._indent()+phrase+calls).lower()
            answer = dflt if i == '' else i[0]
        return answer
    
    def _oco_entries(self,word,cant):
        """Blah blah blah."""
        numb = None
        while not isinstance(numb,(float,int)):
            try:
                numb = eval(input(self._indent()+'Enter the '+word+': '))
            except:
                self._dyn_print(cant,amt=0)
        return numb

    def _c_only_entries(self,dex,cant):
        """Blah blah blah."""
        numb = None
        while not isinstance(numb,(float,int)):
            state = self._indent()+'Enter the order #{:} coefficient'.format(dex)
            dflt = ': ' if dex == 1 else ' (blank to stop): '
            yours = input(state+dflt)
            if (dex >= 2) and (yours == ''):
                numb = '_stop'
                break
            try:
                numb = eval(yours)
            except:
                self._dyn_print(cant)
        return numb

    ### DOUBLE-CHECK THIS METHOD???
    def _vary_rotation_wizard(self,verify_rvL,checked_rv,motions):
        """Blah blah blah."""
        cant = 'Cannot eval, try again.'
        you_tried = len(verify_rvL) > 0
        
        ## Parse the verify list
        if you_tried and np.all(verify_rvL):
            self._dyn_print('RotWizard: your variable rotation is OK. \u2713',amt=0)
            return checked_rv
        
        elif you_tried:
            self._dyn_print('RotWizard: your variable rotation has problems.',amt=1)

            if np.sum(verify_rvL) == 0:
                self._dyn_print('No pieces are formatted correctly:',amt=1)
                self._print_pieces(checked_rv)
                self._dyn_print('Let\'s start over and make a good list.',amt=0)
                wiz_rotval = []
            
            else:
                self._dyn_print('These pieces are formatted wrong:',amt=1)
                badver_rvL = np.logical_not(verify_rvL)
                self._print_pieces(np.asarray(checked_rv,dtype=object)[badver_rvL])

                self._dyn_print('But these pieces are good:',amt=1)
                wiz_rotval = list(np.asarray(checked_rv,dtype=object)[verify_rvL])
                self._print_pieces(wiz_rotval)

                if verify_rvL[0] and (len(wiz_rotval) >= 2):
                    answer = self._ask_question('n','Use only the good pieces and exit the wizard?')
                    self._shift_tab(amt=1)
                    if answer == 'n':
                        self._dyn_print('OK, let\'s modify the rotation more.',amt=-1)
                    else:
                        self._dyn_print('Got it, keeping just the good pieces.',amt=-2)
                        return wiz_rotval
                else:
                    self._dyn_print('Let\'s continue modifying the rotation.',amt=0)

        else:
            self._dyn_print('RotWizard: let\'s create a new variable rotation.',amt=1)
            wiz_rotval = []
        
        ## Reminder
        self._dyn_print('Remember your entries are for *motions* = {:}.'.format(motions),amt=0)

        ## Needs a constant?
        if not you_tried or not verify_rvL[0]:
            self._dyn_print('Constant term is a float or integer, or a string to use last value.',amt=1)
            const = None
            while not isinstance(const,(float,int,str)):
                try:
                    const = eval(input(self._indent()+'Enter a constant (quotes for string): '))
                except:
                    self._dyn_print(cant,amt=0)
            wiz_rotval.insert(0,const)
            self._shift_tab(amt=-1)

        ## Want variations?
        word = 'a' if len(wiz_rotval) == 1 else 'another'
        answer = self._ask_question('y','Add '+word+' variation?')
        more_ki = True if answer == 'y' else False
        if not more_ki:
            self._shift_tab(amt=-1)

        ## Variations loop
        while more_ki:
            self._shift_tab(amt=1)
            ## Kind
            self._dyn_print('Kinds of variation are {:}.'.format(self._accept_rotval),amt=1)
            kind = None
            while kind not in self._accept_rotval:
                kind = input(self._indent()+'Enter a kind (no quotes): ')
            rv_piece = [kind]
            self._shift_tab(amt=-1)
            
            ## Entry style
            self._dyn_print('Entries are either order--coefficient--offset (CO2) or coefficient only.',amt=1)
            answer = self._ask_question('n','Do you want to use CO2 style?')
            co2 = False if answer == 'n' else True

            more_el,dex = True,1

            ## Elements loop
            while more_el:
                self._shift_tab(amt=1)
                if co2:
                    self._dyn_print('Element #{:}'.format(dex),amt=1)
                    order = self._oco_entries('order',cant)
                    coeff = self._oco_entries('coefficient',cant)
                    offset = self._oco_entries('offset',cant)
                
                    rv_piece.append([order,coeff,offset])
                
                    answer = self._ask_question('n','Add another element?')
                    if answer == 'n':
                        more_el = False
                    amt = -2 if more_el else -4
                
                else:
                    coeff = self._c_only_entries(dex,cant)
                    if isinstance(coeff,str):
                        more_el = False
                    else:
                        rv_piece.append(coeff)
                    amt = -1 if more_el else -3
                
                dex += 1
                self._shift_tab(amt=amt)
        
            wiz_rotval.append(rv_piece)
            
            ## More variations?
            answer = self._ask_question('n','Add another variation?')
            if answer == 'n':
                more_ki = False
                self._shift_tab(amt=-1)

        self._dyn_print('RotWizard finished creating your variable rotation!',amt=0)
        return wiz_rotval
    
    def _invert_end_rot_motion(self,motions):
        """Blah blah blah."""
        mot_style,mot_qual = motions[:-1],motions[-1]
        wr = np.atleast_1d(self.Wrot)[-1]
        # So you could always vary Prot later if you want.
        pr = (360*100)*self.Porb if wr == 0 else np.atleast_1d(self.Prot)[-1]
        
        if mot_style == 'freq':  # Converted from rad/second to degrees/day
            if mot_qual == 'A':
                rotval = np.degrees(wr*self.sec_per_day)
            elif mot_qual == 'R':
                rotval = wr*(self.Porb/(2.0*pi))/self._ecc_factor

        elif mot_style == 'per':  # Converted from seconds to days
            sw = 1 if wr >= 0 else -1
            if mot_qual == 'A':
                rotval = (sw*pr)/self.sec_per_day
            elif mot_qual == 'R':
                rotval = (sw*pr)*(self._ecc_factor/self.Porb)
        
        return rotval
    
    
    def _has_param_changed(self,old,new):
        """Blah blah blah."""
        if isinstance(old,np.ndarray) or isinstance(new,np.ndarray):
            test = '_no' if np.array_equal(old,new) else True
        else:
            test = True if old != new else '_no'
        return test
    
    def _check_multi_updater(self,things):
        """Blah blah blah."""
        return any(x != '_no' for x in things)
    
    
    def _calc_orb_period(self):
        return 2.0*pi*(((self.smaxis**3.0)/(self.Mstar*self.grav_const))**0.5)

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
        timeval = np.linspace(t_start,t_end,N+1)

        n_end = n_start + self.numOrbs
        trackorbs = np.linspace(n_start,n_end,N+1)
        return timeval,trackorbs


    def _setup_the_orbit(self):
        """Ha ha ha."""
        # The KeplerEllipse coordinates are: x == "North", y == "East", z == "away from observer".
        # Our orbits are edge-on with inclination = 90 degrees, so orbits in x-z plane.
        # Longitude of ascending node doesn't really matter, so we set Omega = 0 deg (along +x axis).
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
        new_trackorbs = np.linspace(n_start,n_end,N+1)
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
    
    
    def _alter_rot_times(self,_makenew,_reset):
        """Blah blah blah."""
        if _makenew:
            spin_history,timeval_rot = 0,np.copy(self.timeval)
        else:
            t_i = -1 if self._has_T_evolved else 0
            spin_history = self._net_zero_long[t_i]
            if _reset:
                timeval_rot = np.copy(self.timeval - self.timeval[0])
            else:
                # Mimic current timeval shape
                cur_start,cur_end = self.timeval_rot[[0,-1]]
                new_N = self.timeval.size
                timeval_rot = np.linspace(cur_start,cur_end,new_N)
        return spin_history,timeval_rot

    def _rotation_builder(self,rotval):
        """Blah blah blah."""
        t_norm = self.timeval_rot/self.Porb  # seconds to number of orbits
        
        # Factor for logistic function (LF) when only coeffs are passed.
        #     Strictly, standard LF = 0 only when t --> -inf. But this method:
        #     removes the first 1/(1+fac) of growth so LF(t=0) = 0,
        #     divides by fac/(1 + fac) so LF(t-->inf) = 1.0,
        #     and then multiplies LF by coeff.
        # A larger np.log(fac) means steeper LF; fac = 99 gives a nice curve.
        fac = 99
        
        RV_built = rotval[0]  # Constant term
        
        for rv in rotval[1:]:
            kind = rv[0]
            rvA = np.asarray(rv[1:])
            
            if rvA.ndim == 2:  # [[order,coeff,offset],[...],...]
                order = rvA[:,0,np.newaxis]
                coeff = rvA[:,1,np.newaxis]
                offset = rvA[:,2,np.newaxis]
            elif rvA.ndim == 1:  # [coeff1,coeff2,...]
                coeff = rvA[:,np.newaxis]
                seq = np.arange(1,len(rvA)+1)[:,np.newaxis]
                if kind == 'logist':
                    offset = seq/2  # Midpoint of 1 orbit, 2 orbits, etc.
                    order = np.log(fac)/offset
                else:
                    order = seq
                    offset = 0
    
            if kind == 'time':
                alter_RV = (t_norm - offset)**order
            elif kind in ['periS','periC']:
                fun = np.sin if kind[-1] == 'S' else np.cos
                alter_RV = fun(order*(self.tru_anom - np.radians(offset)))
            elif kind in ['phaseS','phaseC']:
                fun = np.sin if kind[-1] == 'S' else np.cos
                alter_RV = fun(order*(np.radians(self.alpha - offset)))
            elif kind == 'logist':
                alter_RV = 1.0/(1 + np.exp(-order*(t_norm - offset)))
                if rvA.ndim == 1:
                    alter_RV -= 1.0/(1 + fac)  # start goes to 0
                    alter_RV /= fac/(1 + fac)  # end limit goes to 1.0
            else:
                alter_RV = np.zeros(coeff.shape)
            
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
        return Prot,Wrot,Wadvec,RV_built
    
    
    def _using_neworold_param(self,new,old):
        """Blah blah blah."""
        return (new,new) if self._check_single_updater(new) else (old,old)
    
    
    def _setup_radiate_recirc(self,tau_rad,epsilon):
        """Blah blah blah."""
        can_set_check = (self.eccen == 0) and isinstance(self.Wrot,(float,int))
        
        if (epsilon != None) and (not np.isnan(epsilon)):
            
            if can_set_check:
                recirc_effic = epsilon
                
                if self.Wadvec == 0:
                    if recirc_effic != 0:
                        self._dyn_print('Constructor warning: atmosphere\'s advective frequency is 0, *recirc_effic* is not.',amt=1)
                        self._dyn_print('Your planet has no winds, but you want to transport heat.',amt=0)
                        self._dyn_print('I am setting radiative time to infinity, but your system is not self-consistent.',amt=-1)
                        print('')
                        radiate_time = np.inf
                    else:
                        radiate_time = 0
            
                else:
                    radiate_time = abs(recirc_effic/self.Wadvec)
                    # Check for mismatched wind direction
                    if abs(np.sign(recirc_effic)-np.sign(self.Wadvec)) == 2:
                        self._dyn_print('Constructor warning: atmosphere\'s advective frequency and *recirc_effic* have opposite signs.',amt=1)
                        self._dyn_print('Your planet\'s winds flow one way, but you want them flowing the other way.',amt=0)
                        self._dyn_print('Radiative time is defined, but your system is not self-consistent.',amt=-1)
                        print('')

            else:
                self._dyn_print('Constructor ignore: you can only set *recirc_effic* if the atmosphere\'s advective frequency is constant.',amt=1)
                if self.eccen != 0:
                    self._dyn_print('Your planet\'s orbit is not circular (orbital angular velocity varies).',amt=0)
                if not isinstance(self.Wrot,(float,int)):
                    self._dyn_print('Your planet\'s spin is not constant (rotational angular velocity varies).',amt=0)
                self._shift_tab(amt=-1)
                print('')
                recirc_effic = np.nan
                radiate_time = tau_rad*self.sec_per_hour  # Converted from hours to seconds
            
        else:
            radiate_time = tau_rad*self.sec_per_hour  # Converted from hours to seconds
            recirc_effic = self.Wadvec*radiate_time if can_set_check else np.nan
        
        return radiate_time,recirc_effic
    
    
    def _setup_colatlong(self,NSIDE):
        """Blah blah blah."""
        colat,longs = hp.pix2ang(NSIDE,list(range(hp.nside2npix(NSIDE))))
        pixel_sq_rad = hp.nside2pixarea(NSIDE)
        on_equator = (colat == (pi/2))
        return colat,longs,pixel_sq_rad,on_equator,NSIDE
    
    def _calc_longs(self):
        """Blah blah blah."""
        # Planet coordinates: longitude = 0 always points at star.
        # Longitude of gas parcels change throughout orbit. Colatitude stays the same.
        # So: new_longs = orig_longs +- Rotation effect (East/West) - Orbit effect (West)
        delta_times = np.ediff1d(self.timeval_rot,to_begin=0)
        net_long_change = np.cumsum(self.Wadvec*delta_times) + self.spin_history
        ## OLD VERSION: net_long_change = (self.Wrot*self.timeval_rot) - self.tru_anom + self.spin_history
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
            infinite_eps = np.atleast_1d(self.Wadvec)[0]*self.radiate_time
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
    
    
    def _parameter_pipeline(self,Teff,Rstar,Mstar,
                            Rplanet,smaxis,eccen,arg_peri,bondA,
                            motions,calc_orb,orbval,rotval,
                            radiate_time,recirc_effic,
                            numOrbs,stepsPerOrb,NSIDE,_makenew):
        """Lots of stuff and things."""
        upd_mot,upd_cal,upd_obv,upd_rtv,upd_Po,upd_tv,upd_kE,upd_Pr,upd_Wadv = self._downpipe_assume_same(9)
        
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
        
        ### We add 180 deg to the input *arg_peri*. See the method here and DON'T GET CONFUSED! :-)
        if self._check_single_updater(arg_peri):
            self.arg_peri = self._modify_arg_peri(arg_peri)
        
        if self._check_multi_updater([motions,calc_orb,orbval,rotval]):
            mots_loc,calo_loc,orbv_loc,rotv_loc = self._parse_motion(motions,calc_orb,orbval,rotval)
            upd_mot = self._has_param_changed(self._last_motions,mots_loc)
            upd_cal = self._has_param_changed(self._last_calc_orb,calo_loc)
            upd_obv = self._has_param_changed(self._last_orbval,orbv_loc)
            # Send to RotWizard?
            if isinstance(rotv_loc,list):
                verify_rvL = self._verify_variable_rotval(rotv_loc)
                rotv_loc = self._vary_rotation_wizard(verify_rvL,rotv_loc,mots_loc)
                print('')  # For gap in prints
                # Are you grabbing a constant term from the previous rotval?
                if isinstance(rotv_loc[0],str):
                    if not _makenew and (upd_mot == True):
                        rotv_loc[0] = self._invert_end_rot_motion(mots_loc)
                    else:
                        rotv_loc[0] = np.atleast_1d(self._last_RV_built)[-1]
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
            # Is the rotation either just changed or not varying?
            _reset = self._check_single_updater(upd_rtv) or (not isinstance(rotv_loc,list))
            self.spin_history,self.timeval_rot = self._alter_rot_times(_makenew,_reset)
            self._should_add_rottime = (not _makenew) and (not _reset)
            
            old_Prot,old_Wadvec = ('_null','_null') if _makenew else (self.Prot,self.Wadvec)
            self.Prot,self.Wrot,self.Wadvec,self._last_RV_built = self._setup_rot_motion(mots_loc,rotv_loc)
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
    
    
    def _setup_lasts(self):
        """Blah blah blah."""
        self._last_radiate_time = '_null'
        self._last_recirc_effic = '_null'
        self._last_motions = 'perR'
        self._last_calc_orb = True
        self._last_orbval = 1.0
        self._last_rotval = 1.0
        self._last_RV_built = 1.0
        return


    def __init__(self,name='Hot Jupiter',Teff=5778,Rstar=1.0,Mstar=1.0,
                 Rplanet=1.0,smaxis=0.1,eccen=0,arg_peri=0,bondA=0,
                 motions='perR',calc_orb=True,orbval=1.0,rotval=1.0,
                 radiate_time=12.0,recirc_effic=None,
                 numOrbs=3,stepsPerOrb=3600,NSIDE=8):
        
        """Some initial constructor stuff.

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
            Argument at periastron in degrees - angle betwen ascending node and periastron and in degrees

        A : Bond albedo
            Setting albedo to a different value reduces incoming flux by a certain fraction.
        
        Porb (float): orbital period in days 
            can be calculated from Kepler's laws
            
        Prot :
            Rotational period of the gas around the planet.
  
        Wadvec :
            Net rate of atmosphere circulation.
 
        T0 :
            Initial temperature of gas at substellar point at periastron.

        tau_rad (and epsilon):
            epsilon = tau_rad * Wadvec

        stepsPerOrb :
            time steps per orbit of planet
            
        numOrbs (int) :
            Number of orbital periods we will integrate for.
        
        NSIDE :
            healpix number of pixels for the planet surface
            NPIX = 12*(NSIDE**2) (ex: 192 pixels --> NSIDE = 4; 798 pixels --> NSIDE = 8)
            
        Quantities attached to instance
        -------------------------------
        
        timeval,timeval_rot -time array (1D)
                
        radius -orbital separation array
        
        Worb - orbital angular velocity array
        
        alpha - phase angle array
        
        frac_lit - illuminted fraction array
        
        colat,longs - initial pixel coordinates for gas
        
        """
        self._tab_level = 0
        
        self._dyn_print('Constructing model...',amt=1)

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
        
        self._shift_tab(amt=-1)
        self._dyn_print('Finished building {:}'.format(self.name),amt=None)
        print(self._method_delim)
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
        if isinstance(self.Prot,(float,int)):
            disp_prot = self.Prot/self.sec_per_day
            slot_pr = '{:^14.2f}'
        else:
            low_pr_d,high_pr_d = np.amin(self.Prot)/self.sec_per_day,np.amax(self.Prot)/self.sec_per_day
            disp_prot = '{:.2f} - {:.2f}'.format(high_pr_d,low_pr_d)
            slot_pr = '{:^14}'
        form_cols = '{:^14.2f} '+slot_pr+' {:^14.3f} {:^24.1f}'
        print(form_cols.format(self.Porb/self.sec_per_day,disp_prot,self.eccen,self.arg_peri))
        print('')
        
        # Wadvec, radiate_time, recirc_effic
        form_cols = '{:^26} {:^22} {:^10}'
        print(form_cols.format('Advective freq. (deg/hr)','Radiative time (hrs)','Epsilon'))
        low_wa_ds = np.degrees(np.amin(self.Wadvec))*self.sec_per_hour
        high_wa_ds = np.degrees(np.amax(self.Wadvec))*self.sec_per_hour
        if low_wa_ds == high_wa_ds:
            disp_wadv = low_wa_ds
            slot_wa = '{:^26.3f}'
        else:
            disp_wadv = '{:.3f} - {:.3f}'.format(low_wa_ds,high_wa_ds)
            slot_wa = '{:^26}'
        form_cols = slot_wa+' {:^22.3f} {:^10.3f}'
        print(form_cols.format(disp_wadv,self.radiate_time/self.sec_per_hour,self.recirc_effic))

        print(self._method_delim)
        return


    ### Changing the system
    def SmartModify_Params(self,name='_no',Teff='_no',Rstar='_no',Mstar='_no',
                           Rplanet='_no',smaxis='_no',eccen='_no',arg_peri='_no',bondA='_no',
                           motions='_no',calc_orb='_no',orbval='_no',rotval='_no',
                           radiate_time='_no',recirc_effic='_no',
                           numOrbs='_no',stepsPerOrb='_no',NSIDE='_no'):
        """Change your stuff around!"""
        self._shift_tab(amt=None)
        
        self._dyn_print('Starting smart mods...',amt=1)
        if self._check_single_updater(name):
            self.name = name
        
        self._parameter_pipeline(Teff,Rstar,Mstar,
                                 Rplanet,smaxis,eccen,arg_peri,bondA,
                                 motions,calc_orb,orbval,rotval,
                                 radiate_time,recirc_effic,
                                 numOrbs,stepsPerOrb,NSIDE,_makenew=False)

        self._shift_tab(amt=-1)
        self._dyn_print('Finished modifying {:}'.format(self.name),amt=None)
        print(self._method_delim)
        return
    
    
    ### Variable Rotation
    
    def RotWizard(self,motions=None):
        """Make some stuff!"""
        self._shift_tab(amt=None)
        
        use_mots = motions if motions in self._accept_motions else self._last_motions
        verify_rvL,checked_rv = [],[]
        
        wiz_rotval = self._vary_rotation_wizard(verify_rvL,checked_rv,use_mots)
        self._shift_tab(amt=None)
        print(self._method_delim)
        return wiz_rotval
    
    def _daybased_convert(self,use_periods,per,freq):
        """Blah blah blah."""
        if use_periods:
            wp = per/self.sec_per_day
        else:
            wp = np.degrees(freq)*self.sec_per_day
        
        if not isinstance(wp,np.ndarray):
            wp *= np.ones(self.timeval_rot.shape)
        return wp
    
    ### PICK UP HERE, LOOKING GOOD! ANY MORE ADD-ONS?
    def HowRotChanges(self,use_periods=False,include_orb=True,include_advec=False,
                      spike_limit=None,mark_zero=True,_combo=False,_axuse=None):
        """Blah blah blah."""
        if _combo:
            axhow = _axuse
        else:
            fig_howrot,axhow = plt.subplots(figsize=(7,7))
        
        rel_time = self.timeval_rot/self.Porb
        vert = 'Period (days)' if use_periods else 'Angular Frequency (degrees / day)'
        c_ro = '0.75' if include_advec else 'k'
        
        wp_rots = self._daybased_convert(use_periods,self.Prot,self.Wrot)
        axhow.plot(rel_time,wp_rots,c=c_ro,lw=2,zorder=3,label='Rotational')
        
        wp_orbs = self._daybased_convert(use_periods,self.Porb,self.Worb)
        if include_orb:
            axhow.plot(rel_time,wp_orbs,c=c_ro,ls='--',lw=1.5,zorder=2,label='Orbital')
        
        make_p = lambda w: np.inf if w == 0 else 2.0*pi/abs(w)
        w_to_p = np.vectorize(make_p) if isinstance(self.Wadvec,np.ndarray) else make_p
        wp_advs = self._daybased_convert(use_periods,w_to_p(self.Wadvec),self.Wadvec)
        if include_advec:
            axhow.plot(rel_time,wp_advs,c='k',ls='-.',lw=2,zorder=4,label='Advective')
        
        if mark_zero:
            axhow.axhline(0,c='0.5',ls=':',zorder=1)
        
        if isinstance(spike_limit,(float,int)):
            test_wp = wp_advs if include_advec else wp_rots
            lo,hi = np.amin(test_wp),np.amax(test_wp)
            cutoff = spike_limit*np.amax(np.absolute(wp_orbs))
            if use_periods:
                f = 0.05  # Padding factor
                v_low,v_high = (-f*cutoff,cutoff) if hi > cutoff else (None,None)
            else:
                v_low = -cutoff if lo < -cutoff else None
                v_high = cutoff if hi > cutoff else None

            axhow.set_ylim(v_low,v_high)

        axhow.set_title('Variable Motion of '+self.name)
        axhow.set_xlabel('Relative Time (orbits)')
        axhow.set_ylabel(vert)

        axhow.legend(loc='best')
        
        if not _combo:
            fig_howrot.tight_layout()
            self.fig_howrot = fig_howrot
            plt.show()
        return
    
    
    ### Draw orbit
    
    def _orbit_auscale(self,pos):
        return pos/self.astro_unit
    
    def _orbit_scatter(self,axorb,pos,ol_sty):
        au_pos = self._orbit_auscale(pos)
        color,mark,ize,lab = ol_sty
        # Brackets on color so e.g. '0.8' is parsed OK.
        axorb.scatter(au_pos[0],au_pos[2],c=[color],marker=mark,s=ize,edgecolors='k',zorder=2,label=lab)
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
        
        if self.numOrbs < 1.0:
            dummy_time = np.linspace(0,self.Porb,self.stepsPerOrb+1)
            au_pos = self.kep_E.xyzPos(dummy_time)/self.astro_unit
        else:
            i_one = self.stepsPerOrb + 1
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
            self.trackorbs += self.numOrbs
            
            (self.radius,self.orb_pos,self.tru_anom,
             self.Worb,self.alpha,self.frac_litup) = self._calc_orbit_props()
        
        # Rotational
        if self._should_add_rottime:
            back_to_zero = False  # Assume at first
            
            if isinstance(self._last_rotval,list):
                self._dyn_print('You are varying the rotation rate of your planet, with *motions* = {:}.'.format(self._last_motions),amt=1)
                self._dyn_print('The current *rotval* is {:}.'.format(self._last_rotval),amt=0)
                self._dyn_print('Or, you can hold the rotation rate at its last evolved value.',amt=0)
                answer = self._ask_question('y','Do you want to keep varying the spin?')
                keep_varying = True if answer == 'y' else False

                some_time = any(r[0] == 'time' for r in self._last_rotval[1:])
                some_logi = any(r[0] == 'logist' for r in self._last_rotval[1:])
                if keep_varying and (some_time or some_logi):
                    print('')
                    if some_time:
                        self._dyn_print('Your variable *rotval* has explicit \'time\' components.',amt=1)
                        self._dyn_print('It could give bad rotation rates if your planet evolves long enough.',amt=0)
                    else:
                        self._dyn_print('Your variable *rotval* has \'logist\' components.',amt=1)
                        self._dyn_print('They will be constants once your planet evolves long enough.',amt=0)
                    answer = self._ask_question('n','Do you want to reset all components in *rotval* to t=0?')
                    back_to_zero = False if answer == 'n' else True
                    self._shift_tab(amt=-2)
                elif not keep_varying:
                    self._shift_tab(amt=-1)
                    ## THINK THIS IS OK BUT DOUBLE-CHECK???
                    self._last_rotval = self._last_RV_built[-1]  # Change gets passed below
                
                print('')
            
            if back_to_zero:
                self.spin_history,self.timeval_rot = self._alter_rot_times(_makenew=False,_reset=True)
            else:
                # Probably don't need to check _has_T_evolved: you shouldn't be here unless it has.
                # But I'm leaving it for now, just in case. Can't hurt. :-)
                t_i = -1 if self._has_T_evolved else 0
                self.spin_history = self._net_zero_long[t_i]
                self.timeval_rot += self.Porb*self.numOrbs

        # Advective
        if self._should_add_orbtime or self._should_add_rottime:
            self.Prot,self.Wrot,self.Wadvec,self._last_RV_built = self._setup_rot_motion(self._last_motions,
                                                                                         self._last_rotval)
            
            self.radiate_time,self.recirc_effic = self._setup_radiate_recirc(self._last_radiate_time,
                                                                             self._last_recirc_effic)

            self.longs_evolve,self._net_zero_long = self._calc_longs()
            
            (self.illumination,self.visibility,
             self.SSP_long,self.SOP_long) = self._calc_vis_illum()

        return


    def _diff_eq_tempvals(self,start_Tvals):
        """Something something else."""
        Tvals_evolve = np.zeros(self.longs_evolve.shape)
        Tvals_evolve[0,:] += start_Tvals
        
        # Check for circular orbit + constant rotational frequency
        if isinstance(self.Wadvec,(float,int)):
            if abs(self.recirc_effic) <= 10**(-4):
                Tvals_evolve = ((1.0-self.bondA)*self.illumination)**(0.25)
            else:
                # Advective frequency is constant so sign spcifies direction atmosphere rotates.
                the_sign = -1.0 if self.Wadvec < 0 else 1.0
                delta_longs = np.diff(self.longs_evolve,n=1,axis=0) % (the_sign*2.0*pi)

                for i in range(1,self.timeval.size):
                    # Stellar flux is constant for circular orbits, F(t)/Fmax = 1.
                    delta_Tvals = (1.0/self.recirc_effic)*(self.illumination[i-1,:] - (Tvals_evolve[i-1,:]**4))*delta_longs[i-1,:]
                    Tvals_evolve[i,:] = Tvals_evolve[i-1,:] + delta_Tvals  # Step-by-step T update
    
        else:
            # Normalized stellar flux
            if self.eccen != 0:
                scaled_illum = self.illumination*((self.smaxis*(1-self.eccen)/self.radius[:,np.newaxis])**2)
            else:
                scaled_illum = self.illumination  # Another circular orbit
            
            # Eccentric DE uses t_tilda = t/radiate_time
            if self.radiate_time <= 10**(-4):
                Tvals_evolve = ((1.0-self.bondA)*scaled_illum)**(0.25)
            else:
                delta_radtime = np.ediff1d(self.timeval)/self.radiate_time

                for i in range(1,self.timeval.size):
                    delta_Tvals = (scaled_illum[i-1,:] - (Tvals_evolve[i-1,:]**4))*delta_radtime[i-1]
                    Tvals_evolve[i,:] = Tvals_evolve[i-1,:] + delta_Tvals  # Step-by-step T update
        
        return Tvals_evolve
    
    ## ADD WAY TO AUTOMATE EVOLVING UNTIL SOME EQUILIBRIUM IS REACHED? ##
    def Evolve_AtmoTemps(self):
        """Something something else."""
        self._shift_tab(amt=None)
        
        t_i,s = (-1,'Re-heating') if self._has_T_evolved else (0,'Heating')
        start_Tvals = self.Tvals_evolve[t_i,:]
        
        self._dyn_print(s+' {:}...'.format(self.name),amt=1)
        self._update_params_before_evolve()
        o_start,o_end = self.trackorbs[[0,-1]]
        self._dyn_print('...Moving through orbits {:.2f} to {:.2f}...'.format(o_start,o_end),amt=-1)

        self.Tvals_evolve = self._diff_eq_tempvals(start_Tvals)
        self._has_T_evolved = True
        self._should_add_orbtime = True
        self._should_add_rottime = True
        
        self._dyn_print('Evolving complete',amt=None)
        print(self._method_delim)
        return
    
    
    ### Temperature Map
    
    def _can_make_the_fig(self,which,_extra):
        """Something something else."""
        quit_out = False
        if not self._has_T_evolved and not _extra:
            self._dyn_print('Data Flag: You have not baked '+self.name+' since mixing in some new parameters.',amt=1)
            if self.Tvals_evolve.shape[0] == self.timeval.size:
                self._dyn_print('I am using the calculated temperatures from before, but they may not be accurate now.',amt=0)
            else:
                quit_out = True
                self._dyn_print('There are no calculated temperatures right now, so you can\'t run *'+which+'* yet.',amt=0)
            self._dyn_print('Please run *Evolve_AtmoTemps* and then things should be good. :-)',amt=-1)
        return quit_out
    
    def _final_orbit_index(self):
        """Something something else."""
        # Ensures ending phase is not included twice.
        f_o_i = self.timeval.size - self.stepsPerOrb
        return max(f_o_i,0)
    
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

    def _orth_cbar(self,axbar,new_map,low,high,far_side):
        """Something something else."""
        cb = plt.colorbar(new_map,cax=axbar,orientation='horizontal',ticks=[low,high])
        
        ty = -0.75 if far_side else -1.0
        tx,r = 0.5,'horizontal'
        unit_of_T = r'Normalized Temperature $(T \ / \ T_{0})$'
        cb.ax.text(tx,ty,unit_of_T,rotation=r,fontsize='large',ha='center',va='center',
                   transform=cb.ax.transAxes)
        return
    
    ### ADD FEEDBACK WHEN LOW numOrbs MEANS phase IS BAD???
    def Orth_Mapper(self,phase,relative_periast=False,force_contrast=False,far_side=False,
                    _combo=False,_axuse=None,_cax=None,_i_phase=None):
        """Something something else."""
        quit_out = self._can_make_the_fig('Orth_Mapper',_combo)
        if quit_out:
            return
        
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
        xpix,hsiz,bcut,xval = (2400,11,2,2) if far_side else (1200,7,1,1)
        pic_map = hp.visufunc.orthview(fig=13,map=heat_map,rot=(sop_rot,0,0),flip='geo',
                                       min=low,max=high,cmap=inferno_mod_,
                                       half_sky=not(far_side),xsize=xpix,return_projected_map=True)
        plt.close(13)
        
        if _combo:
            axmap = _axuse
            axbar = _cax
        else:
            fig_orth = plt.figure(figsize=(hsiz,7))
            axmap = plt.subplot2grid((15,hsiz),(0,0),rowspan=14,colspan=hsiz,fig=fig_orth)
            axbar = plt.subplot2grid((15,hsiz),(14,bcut),rowspan=1,colspan=(hsiz-2*bcut),fig=fig_orth)

        new_map = axmap.imshow(pic_map,origin='lower',extent=[-xval,xval,-1,1],
                               vmin=low,vmax=high,cmap=inferno_mod_)
        self._orth_graticule(axmap,zero_to_sop,far_side)

        ### Have my custom graticule now, but keeping this for posterity:
        # Seems like graticule + orthview can throw out two invalid value warnings.
        # Both pop up when *half_sky* is True, only one when it's False.
        # Then again, if the numbers in *rot* have > 1 decimal place,
        # those warnings can disappear! It's something weird in healpy's
        # projector.py and projaxes.py. I'm suppressing both warnings for now.
        #### with warnings.catch_warnings():
        ####     warnings.filterwarnings('ignore',message='invalid value encountered in greater')
        ####     hp.visufunc.graticule(local=True,verbose=True)

        if relative_periast:
            descrip = r' $%.2f^{\circ}$ from periastron' % (np.degrees(self.tru_anom[i_want]) % 360)
        else:
            descrip = r' at $%.2f^{\circ}$ orbital phase' % self.alpha[i_want]
        axmap.set_title(self.name + descrip)
        self._orth_cbar(axbar,new_map,low,high,far_side)

        if far_side:
            axmap.text(-1,-1.08,'Observer Side',size='large',ha='left',va='center')
            axmap.text(1,-1.08,'Far Side',size='large',ha='right',va='center')

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
    
    
    def _combo_faxmaker(self,figsize,srow,scol,bcut):
        """Blah blah blah."""
        f_com = plt.figure(figsize=figsize)
        spec = (srow,2*scol)
        _axl = plt.subplot2grid(spec,(0,0),rowspan=srow,colspan=scol,fig=f_com)
        _axr = plt.subplot2grid(spec,(0,scol),rowspan=(srow-1),colspan=scol,fig=f_com)
        _cax = plt.subplot2grid(spec,(srow-1,scol+bcut),rowspan=1,colspan=(scol-2*bcut),fig=f_com)
        return f_com,_axl,_axr,_cax

    def Combo_OrbitOrth(self,phase,relative_periast=False,show_legend=True,force_contrast=False):
        """Blah blah blah."""
        quit_out = self._can_make_the_fig('Combo_OrbitOrth',False)
        if quit_out:
            return
        
        fig_orborth,_axorb,_axmap,_cax = self._combo_faxmaker(figsize=(14,7),srow=15,scol=7,bcut=1)

        # Return phase position before drawing orbit
        _phxyz = self.Orth_Mapper(phase,relative_periast,force_contrast,far_side=False,
                                  _combo=True,_axuse=_axmap,_cax=_cax)
        
        self.Draw_OrbitOverhead(show_legend,_combo=True,_axuse=_axorb,_phxyz=_phxyz)
            
        fig_orborth.tight_layout(w_pad=0)
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
                      kind='obs',run_integrals=False,bolo=False,separate=False,_extra=False):
        """Blah blah blah."""
        quit_out = self._can_make_the_fig('Observed_Flux',_extra)
        if quit_out:
            null = np.zeros(self.timeval.shape)
            return (null,null) if separate else null
        
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
        bad_begin = False
        # Ensures ending phase is not included twice.
        fin_orb_start = self._final_orbit_index()
        
        if fin_orb_start == 0:
            i_start,i_end = 0,self.timeval.size
            bad_begin = True
        else:
            # The possible *begins*, with (fun trig angle) checks:
            # periast (xct), apast (nct), transit (xca), eclipse (nca), ascend (xsa), descend (nsa)
            fun = np.argmax if begins in ['periast','transit','ascend'] else np.argmin
            trig = np.sin if begins in ['ascend','descend'] else np.cos
            angle = self.tru_anom[fin_orb_start:] if begins in ['periast','apast'] else np.radians(self.alpha[fin_orb_start:])
            fi_end = fun(trig(angle))

            if (fi_end + fin_orb_start) < self.stepsPerOrb:
                # Can't use this *begins* so final orbit instead.
                fi_end = self.stepsPerOrb - 1
                bad_begin = True
            
            i_end = fi_end + fin_orb_start
            
            i_start = i_end - self.stepsPerOrb
            i_end += 1  # To have initial phase repeated
        return i_start,i_end,bad_begin

    def _light_times(self,i_start,i_end):
        """Blah blah blah."""
        t_act = self.timeval[i_start:i_end]
        t_start,t_end = t_act[[0,-1]]
        
        o_start = t_start/self.Porb  # For current orbit, not cumulative from *trackorbs*.
        t_rel = self.trackorbs[i_start:i_end] - self.trackorbs[i_start]
        return t_act,t_start,t_end,o_start,t_rel
    
    def _prop_plotter(self,axlig,t_a,t_start,f_terp,ol_sty,y_mark,_combo,_inc):
        """Blah blah blah."""
        f_v = f_terp(t_a)
        t_r = (t_a - t_start)/self.Porb
        
        color = ol_sty[0]
        axlig.plot([t_r,t_r],[0,f_v],c=color,ls='--',zorder=2)
        if _combo:
            mark,ize = ol_sty[1:3]
            lab = ol_sty[-1] if _inc else '_null'
            # Brackets on color so e.g. '0.8' is parsed OK.
            axlig.scatter(t_r,y_mark,c=[color],marker=mark,s=ize,edgecolors='k',zorder=2,label=lab)
        return
    
    def _prop_plotcheck(self,axlig,prop_time,o_start,t_start,t_end,f_terp,
                        ol_sty,y_mark,_combo):
        """Blah blah blah."""
        t_a = prop_time+(np.floor(o_start)*self.Porb)
        _inc = True  # For putting marker in legend
        while t_a <= t_end:
            if t_a >= t_start:
                self._prop_plotter(axlig,t_a,t_start,f_terp,ol_sty,y_mark,_combo,_inc)
                _inc = False  # No double-listed markers!
            t_a += self.Porb
        return
    
    def _light_window(self,axlig,lc_high,f_y,ol_sty,bad_begin):
        """Blah blah blah."""
        axlig.set_ylim(-f_y*lc_high,(1+f_y)*lc_high)
        
        axlig.set_title('Light Curve of '+self.name)
        xl = 'Relative Time' if bad_begin else 'Time from '+ol_sty[-1]
        axlig.set_xlabel(xl+' (orbits)')
        axlig.set_ylabel('Flux ( planet / star )')
        return

    ### ADD FEEDBACK WHEN LOW numOrbs MEANS begins IS BAD???
    def Draw_LightCurve(self,wave_band=False,a_microns=6.5,b_microns=9.5,
                        run_integrals=False,bolo=False,begins='periast',multi_orbit=False,
                        _combo=False,_axuse=None,_phase=None,_relperi=None):
        """Blah blah blah."""
        quit_out = self._can_make_the_fig('Draw_LightCurve',_combo)
        if quit_out:
            return
        
        if begins not in self._accept_begins:
            self._dyn_print('Draw_LightCurve error: strings for *begins* are',amt=1)
            self._dyn_print(self._accept_begins,amt=None)
            plt.close()  # Remove WIP plot if combo method
            return
        
        lightcurve_flux = self.Observed_Flux(wave_band,a_microns,b_microns,
                                             'obs',run_integrals,bolo,False,
                                             _extra=True)
        if _combo:
            axlig = _axuse
        else:
            fig_light,axlig = plt.subplots(figsize=(7,7))
        
        i_start,i_end,bad_begin = self._light_indices(begins)
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
        if multi_orbit:
            m_i = np.array([i_start,i_end]) - self.stepsPerOrb
            while m_i[0] >= 0:
                lcf_mul = lightcurve_flux[m_i[0]:m_i[1]]
                axlig.plot(t_rel,lcf_mul,c='0.67',lw=2,zorder=0)
                m_i -= self.stepsPerOrb
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
        
        self._light_window(axlig,lc_high,f_y,orbloc_styles_[begins],bad_begin)
        
        if not _combo:
            fig_light.tight_layout()
            self.fig_light = fig_light
            plt.show()
        elif _phase != None:
            return _i_phase

        return
    
    
    def Combo_OrbitLC(self,show_legend=True,wave_band=False,a_microns=6.5,b_microns=9.5,
                      run_integrals=False,bolo=False,begins='periast',multi_orbit=False):
        """Blah blah blah."""
        quit_out = self._can_make_the_fig('Combo_OrbitLC',False)
        if quit_out:
            return
        
        fig_orblc = plt.figure(figsize=(14,7))
        
        _axorb = plt.subplot(121)
        self.Draw_OrbitOverhead(show_legend,_combo=True,_axuse=_axorb)
        
        _axlig = plt.subplot(122)
        self.Draw_LightCurve(wave_band,a_microns,b_microns,run_integrals,bolo,
                             begins,multi_orbit,_combo=True,_axuse=_axlig)
        
        fig_orblc.tight_layout(w_pad=2)
        self.fig_orblc = fig_orblc
        plt.show()
        return
    
    
    def Combo_LCOrth(self,phase,relative_periast=False,force_contrast=False,
                     wave_band=False,a_microns=6.5,b_microns=9.5,
                     run_integrals=False,bolo=False,begins='periast',multi_orbit=False,show_legend=True):
        """Blah blah blah."""
        quit_out = self._can_make_the_fig('Combo_LCOrth',False)
        if quit_out:
            return
        
        fig_lcorth,_axlig,_axmap,_cax = self._combo_faxmaker(figsize=(14,7),srow=15,scol=7,bcut=1)
        
        # Return correct phase index to override calc in Orth_Mapper
        _i_phase = self.Draw_LightCurve(wave_band,a_microns,b_microns,run_integrals,bolo,
                                        begins,multi_orbit,_combo=True,_axuse=_axlig,
                                        _phase=phase,_relperi=relative_periast)
        
        # Check if light curve quit with bad *begins*
        if _i_phase == None:
            return
        
        self.Orth_Mapper(phase,relative_periast,force_contrast,far_side=False,
                         _combo=True,_axuse=_axmap,_cax=_cax,_i_phase=_i_phase)
                         
        if show_legend:
            _axlig.legend(loc='best')
      
        fig_lcorth.tight_layout(w_pad=0)
        self.fig_lcorth = fig_lcorth
        plt.show()
        return
    
    
    ### General Combo plots
    def _combo_specs(self,axis,srow,scol):
        """Blah blah blah."""
        rs,cs = srow,scol

        if axis == 'ortho':
            rs -= 1
        elif axis in ['light','motion']:
            l_test = (axis == 'light') and (False)
            m_test = (axis == 'motion') and (False)
            if l_test or m_test:
                cs *= 2

        return rs,cs

    def _new_combo_faxmaker(self,want_axes):
        """Blah blah blah."""
        srow,scol,bcut = 15,7,1

        rs_zero,cs_zero = self._combo_specs(want_axes[0],srow,scol)
        rs_one,cs_one = self._combo_specs(want_axes[1],srow,scol)
        nc = int((cs_zero + cs_one)/scol)

        if nc == 4:
            fsiz = (10,10)
            spec = (2*srow,2*scol)
            loc_one = (rs_zero,0)
        else:
            fpad,cpad = (0,0) if ('ortho' in want_axes) else (1,1)  ### (1,1) seems best
            fsiz = ((nc*7)+fpad,7)
            spec = (srow,(nc*scol)+cpad)
            loc_one = (0,cs_zero+cpad)

        f_com = plt.figure(figsize=fsiz)
        _axzero = plt.subplot2grid(spec,(0,0),rowspan=rs_zero,colspan=cs_zero,fig=f_com)
        _axone = plt.subplot2grid(spec,loc_one,rowspan=rs_one,colspan=cs_one,fig=f_com)
        if 'ortho' in want_axes:
            loc_bar,cs_bar = (((rs_zero,bcut),(cs_zero - 2*bcut)) if want_axes[0] == 'ortho' else
                              ((rs_one,cs_zero+bcut),(cs_one - 2*bcut)))
            _axbar = plt.subplot2grid(spec,loc_bar,rowspan=1,colspan=cs_bar,fig=f_com)
        else:
            _axbar = None

        return f_com,[_axzero,_axone],_axbar
    
    ### temp area: function calls for ComboGraphics
#    LightCurve
#    wave_band=False,a_microns=6.5,b_microns=9.5,
#        run_integrals=False,bolo=False,begins='periast',multi_orbit=False,
#            _combo=False,_axuse=None,_phase=None,_relperi=None
#
#    Orth_Mapper
#    phase,relative_periast=False,force_contrast=False,far_side=False,
#        _combo=False,_axuse=None,_cax=None,_i_phase=None
#
#    OrbitOverhead
#    show_legend=True,_combo=False,_axuse=None,_phxyz=None
#
#    HowRotChange
#    use_periods=False,include_orb=True,include_advec=False,
#        spike_limit=None,mark_zero=True,_combo=False,_axuse=None
    ###

    def _group_combo_kwargs(self,want_axes,**kwargs):
        """Something!"""
        good_kwargs = [None,None]
        
        for i in range(2):
            wax = want_axes[i]
            
            if wax == 'light':
                kw_names = ('wave_band','a_microns','b_microns','run_integrals','bolo','begins','multi_orbit')
            elif wax == 'ortho':
                kw_names = ('relative_periast','force_contrast','far_side')
            elif wax == 'orbit':
                kw_names = ('show_legend','_notarealword_')  # Dummy to loop strings below, not each character!
            elif wax == 'motion':
                kw_names = ('use_periods','include_orb','include_advec','spike_limit','mark_zero')
            
            full_kw = {key:kwargs.get(key,'_null') for key in kw_names}
            good_kwargs[i] = {key:val for key,val in full_kw.items() if val is not '_null'}
        
        return good_kwargs

    ### PICK UP HERE, LOOKING GOOD! NEEDS TWEAKS TO USE ALL THE HIDDEN KEYWORDS.
    def ComboGraphics(self,want_axes=['orbit','light'],**kwargs):
        """Mix and Match!"""
        fig_combo,_axlis,_axbar = self._new_combo_faxmaker(want_axes)
        good_kwargs = self._group_combo_kwargs(want_axes,**kwargs)
        
        phase = kwargs.get('phase',180)  # In case it is not given
        
        dex = lambda nm,wa: 0 if nm == wa[0] else 1
        
        if 'light' in want_axes:
            i = dex('light',want_axes)
            self.Draw_LightCurve(_combo=True,_axuse=_axlis[i],**good_kwargs[i])
        if 'ortho' in want_axes:
            i = dex('ortho',want_axes)
            _phxyz = self.Orth_Mapper(phase,_combo=True,_axuse=_axlis[i],_cax=_axbar,**good_kwargs[i])
        if 'orbit' in want_axes:
            i = dex('orbit',want_axes)
            self.Draw_OrbitOverhead(_combo=True,_axuse=_axlis[i],**good_kwargs[i])
        if 'motion' in want_axes:
            i = dex('orbit',want_axes)
            self.HowRotChanges(_combo=True,_axuse=_axlis[i],**good_kwargs[i])
    
        plt.tight_layout()
        self.fig_combo = fig_combo
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
