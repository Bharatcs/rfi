import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import FK5
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from astropy import units
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate
import copy
import os
import sys
from tqdm.notebook import tqdm
import glob
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = "18"
from utils import *
from constants import *
from sky import SkyModel
from beam import AntennaModel
from scipy.interpolate import RegularGridInterpolator
import h5py

class Observations(object):
        def __init__(self, **kwargs):

            for key, value in kwargs.items():
                setattr(self, key, value)

            self.sky = SkyModel(**self.__dict__)
            self.sky.gen_newGMOSS()
            self.model = AntennaModel(**self.__dict__)

            self.model.get_beam()
            
            if hasattr(self, 'file_list'):   
                self.model.normalize_beam()
            self.model.get_gamma()

            self.interpolate_beam()
            self.interpolate_gamma()
            self.set_location()

        def set_location(self):
            location = EarthLocation(lat=self.SITE_LATITUDE*u.deg, lon=self.SITE_LONGITUDE*u.deg, height=self.ELEVATION*u.m)
            gc       = SkyCoord(l=self.sky.ll_coordinate*u.radian, b=self.sky.bb_coordinate*u.radian,\
                            frame='galactic')
            self.gc  = gc
            self.location = location

        def interpolate_beam(self):
            if all(hasattr(self, attr) for attr in ['file_list', 'gamma_file']):
                beam_func = RegularGridInterpolator((self.model.freq_array, self.model.theta_array, \
                                                     self.model.phi_array), self.model.beam_3D_norm)
            else:
                beam_func = lambda ff : np.cos(np.radians(ff[1]))**2 
            
            self.beam_val = beam_func
            #self.beam_val = self.model.beam_analytic

        def interpolate_gamma(self):
            self.gamma_val = interpolate.interp1d(self.model.gamma_freq, self.model.gamma_val)

        def convolve(self):

            T_bws          = np.zeros((self.obstimes.shape[0], self.sky.freq.shape[0]))

            for val, (time,i) in enumerate(zip(self.obstimes, tqdm(range(len(self.obstimes))))):
        
                print("Processing time: {}".format(time))
           
                #Get Alt, Az
                
                trans_local                 = self.gc.transform_to(AltAz(obstime=time, location=self.location))
                az, alt                     = trans_local.az.degree, trans_local.alt.degree
                
                ind_below_horizon           = alt < 0
                
                beam_gen = np.zeros_like(self.sky.tsky)
                
                ############################
                
                for ifreq, freq_value in enumerate(self.sky.freq):
                    rogue_phi  = []
                    for iangle, (alt_value, az_value) in enumerate(zip(alt, az)):
                        
                        if az_value > self.model.phi_array.max():
                            rogue_phi.append(az_value)
                            az_value = 360 - az_value
                            
                        beam_gen[iangle,ifreq] = self.beam_val([freq_value, alt_value, az_value])
                        
                beam_gen[ind_below_horizon,:] = np.nan
                T_bws[val]                    = np.nansum(self.sky.tsky * beam_gen, axis=0)/np.nansum(beam_gen, axis=0)

            self.tbws = T_bws

        def write_output(self):
            if self.file_name is not None:
                fname = self.file_name
            else:
                fname = 'you_never_provided_a_name'
            f = h5py.File(os.path.join(self.path_output, (fname+'.h5')), 'w')

            grp = f.create_group('index_map')
            grp.create_dataset('frequency', data=self.sky.freq)
            grp.create_dataset('LST', data=self.lst)

            grp2 = f.create_group('ancillary_prod')
            if all(hasattr(self, attr) for attr in ['file_list', 'gamma_file']):
                grp2.create_dataset('beam', data=self.model.beam_3D_norm)
                grp2.create_dataset('ref_eff', data=(1-self.gamma_val(self.sky.freq)**2))

            f.create_dataset('T_A', data = self.ta)
            f.create_dataset('T_A_only_beam', data = self.ta_only_beam)
            f.close()
           
        def make_observations(self):

            self.convolve()
            if self.include_gamma:
                s11 = self.gamma_val(self.sky.freq)
                ref_eff = 1-s11**2
                self.ta =  ref_eff[None,:] * self.tbws 
            else:
                self.ta = self.tbws

            self.ta_only_beam = self.tbws

            observing_time = Time(self.obstimes, scale='utc', location=self.location)
            LST            = observing_time.sidereal_time('mean').value
            self.lst = LST
            self.freq = self.sky.tsky

            if self.output:
                self.write_output()
