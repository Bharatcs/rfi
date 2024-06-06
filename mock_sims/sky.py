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

PATH="sky_models/"


class SkyModel(object):

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def gen_newGMOSS(self):

       
        ll_coordinate, bb_coordinate = np.radians(Read_Two_Column_File(os.path.join(PATH,filename_coord)))
        T_pix_org = np.loadtxt('/home/saurabhs/Documents/gmoss_proto/gmoss_pre/GMOSS_PRATUSH.txt')

        freq  = np.arange(55,110,0.244)
        T_pix = T_pix_org
        self.freq = freq
        self.freq_array = freq
        self.tsky = T_pix
        self.ll_coordinate = ll_coordinate
        self.bb_coordinate = bb_coordinate

    def gen_GMOSS(self):

        ll_coordinate, bb_coordinate = np.radians(Read_Two_Column_File(os.path.join(PATH,filename_coord)))

#         T1, nspec1 = Read_pixel_freq(PATH+filename_pix_spec1, RNUMBER_OF_CHANNELS)
#         T2, nspec2 = Read_pixel_freq(PATH+filename_pix_spec2, QNUMBER_OF_CHANNELS)
#         T3, nspec3 = Read_pixel_freq(PATH+filename_pix_spec3, PNUMBER_OF_CHANNELS)

#         freq    = np.zeros(nspec1+nspec2+nspec3)
#         T_pix   = np.zeros((NHPIX, nspec1+nspec2+nspec3))

#         for i in range(0, nspec2):
#             freq[i]=(float)(QSTART_FREQUENCY/1000.0) +\
#                     ((float)(QCHANNEL_WIDTH)/1000.0)*(float)(i)
#         for i in range(nspec2, (nspec1+nspec2)):
#             freq[i]=(float)(RSTART_FREQUENCY/1000.0) + ((float)(RCHANNEL_WIDTH)/1000.0)*(float)(i-nspec2)
#         for i in range(nspec1+nspec2, (nspec1+nspec2+nspec3)):
#             freq[i]=(float)(PSTART_FREQUENCY/1000.0)+((float)(PCHANNEL_WIDTH)/1000.0)*(float)(i-(nspec1+nspec2))

#         for j in range(0, NHPIX):
#             T_pix[j][0:nspec2]=T2[j]
#             T_pix[j][(nspec2):(nspec1+nspec2)]=T1[j]
#             T_pix[j][(nspec1+nspec2):(nspec1+nspec2+nspec3)]=T3[j]
        T_pix = np.loadtxt('/home/saurabhs/Documents/gmoss_proto/gmoss_pre/GMOSS_PRATUSH.txt')
        freq = np.arange(55,110,0.244)
        freq_org, T_pix_org = copy.deepcopy(freq), copy.deepcopy(T_pix)
        ifr_low, ifr_hgh = select_freq_1d(freq_org, self.fmin, self.fmax)
        freq  = freq_org[ifr_low:ifr_hgh]*1e3
        T_pix = T_pix_org[:,ifr_low:ifr_hgh]

        self.freq = freq
        self.freq_array = freq
        self.tsky = T_pix
        self.ll_coordinate = ll_coordinate
        self.bb_coordinate = bb_coordinate
