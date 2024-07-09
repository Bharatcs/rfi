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
from tqdm import tqdm
import glob
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = "18"
from utils import *
from constants import *
import warnings

def get_freq_from_file(filename):
    #_temp = os.path.basename(filename).replace('.txt','').replace('mhz','')
    _temp = os.path.basename(filename).replace('.txt','').replace('farfield (f=','').split(')')[0] 
    return float(_temp)

class AntennaModel(object):

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def beam_analytic(self,ff):
        '''
        Calling this function in get_beam below
        to interpolate the beam. Used for assessing
        interpolation errors.
        If called from observe.py in interpolate_beam,
        it will be directly computed from given
        theta and freq. 
        '''
        freq = ff[0]
        theta = ff[1]
        freq0, alpha0, sigma0, theta0 = 75, -1, np.radians(20), np.radians(0)
        sigma = sigma0 * (freq/freq0)**(alpha0)
        beam_unnormed = np.exp(-(np.radians(theta))**2/(2*sigma**2))
        return beam_unnormed

    def get_beam(self):
        if hasattr(self, 'file_list'):   
            freq_array  = []
            file_array  = []
            for ii, file_add in enumerate(self.file_list):
                freq_array.append(get_freq_from_file(file_add))
                file_array.append(file_add)

            isort           = np.argsort(np.array(freq_array))
            self.file_array = np.array(file_array)[isort]
            self.freq_array = np.array(freq_array)[isort]

            beam_3D_unnorm = np.zeros((len(self.freq_array), len(self.theta_array), len(self.phi_array)))

            for ii, freq in enumerate(self.freq_array):
                print("Processing frequency {:.1f} MHz".format(freq))
                file_add = self.file_array[ii]
                with open(file_add) as fa:
                    for line_aa in fa.readlines()[2:]:
                        line_aa = line_aa.strip()
                        col1    = line_aa.split('\t')
                        all_val = np.array(list(map(float, col1[0].split())))

                        theta   = 90 - all_val[0]
                        phi     = all_val[1]
                        beam    = all_val[2]
                        iphi    = np.where(self.phi_array==phi)[0]
                        itheta  = np.where(self.theta_array==theta)[0]
                        
                        if theta<=0: #for SARAS, no response below horizon
                            beam = 0.0
                        #beam_3D_unnorm[ii, itheta, iphi] = self.beam_analytic([freq,theta]) 
                        beam_3D_unnorm[ii, itheta, iphi] = beam 

            self.beam_3D_unnorm = beam_3D_unnorm
        else:  
            self.freq_array = self.sky.freq_array # The freq_array is called from SkyModel function

    def normalize_beam(self):
        beam_3D = np.zeros_like(self.beam_3D_unnorm)
        for ifreq in range(beam_3D.shape[0]):
            for iphi in range(beam_3D.shape[2]):
                beam_3D[ifreq,:,iphi] = self.beam_3D_unnorm[ifreq, :, iphi]/np.nanmax(self.beam_3D_unnorm[ifreq, :, iphi])
        self.beam_3D_norm = beam_3D
        print(np.sum(np.isnan(beam_3D)))
    
    def smooth_beam(self):
        fmin = 55
        fmax = 110
        freq_beam_index = np.logical_and((self.freq_array>=fmin),(self.freq_array<=fmax))
        for itheta in range(len(self.theta_array)):
            for iphi in range(len(self.phi_array)):
                if np.sum(self.beam_3D_norm[:,itheta, iphi]==0):
                    continue
                cc = np.polyfit(self.freq_array[freq_beam_index], self.beam_3D_norm[freq_beam_index,itheta, iphi],6)
                ff = np.polyval(cc, self.freq_array[freq_beam_index])
                self.beam_3D_norm[freq_beam_index,itheta, iphi] = ff
                self.beam_3D_norm[np.logical_not(freq_beam_index),itheta,iphi]=0.0
    
    def get_gamma(self):
        if hasattr(self, 'gamma_file'):
            gamma_freq = []
            gamma_val = []
            with open(self.gamma_file) as fa:
                for line_aa in fa.readlines()[3:]:
                    line_aa = line_aa.strip()
                    col1    = line_aa.split('\t')
                    _freq   = np.array(list(map(float, col1[0].split())))[0]
                    _val    = 10**((np.array(list(map(float, col1[1].split())))[0])/20)
                    gamma_freq.append(_freq)
                    gamma_val.append(_val)          
            self.gamma_freq = np.array(gamma_freq)
            self.gamma_val  = np.array(gamma_val)
        else:
            a = np.linspace(10,305,1001)
            b = np.zeros(1001) 
            self.gamma_freq = np.array(a)
            self.gamma_val  = np.array(b)
