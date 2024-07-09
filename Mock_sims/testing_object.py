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
from observe_v3 import Observations

import time
start = time.time()

output_name = "tcg_on_raft"

SITE_LATITUDE       = 32.81444444 #/* +32.77944444   deg for Hanle */ 
SITE_LONGITUDE      = 78.85555556 #/* 78.96416667 deg for Hanle */
ELEVATION = 4500

dt = TimeDelta(np.linspace(0.,24.*3600, 500), format='sec') #Will NOT take ~5 min
obstimes = Time('2022-10-15 00:00:00') + dt

fmin = 40/1e3
fmax = 110/1e3
phi_res   = 1
theta_res = 1

phi_array   = np.arange(0, 360, phi_res)
theta_array = np.arange(-90, 90 + theta_res, theta_res)

beam_path  = '/Users/EoR/Desktop/SARAS_3/saras3_land/saras_floating_over_layered_soil_layer1_eps_5_ht_3.7m/'
rt_file    = os.path.join(beam_path,'s11_linear.txt')
beam_list  = sorted(glob.glob(os.path.join(beam_path,"farfield*.txt")))
#beam_list  = sorted(glob.glob(os.path.join(beam_path,"*mhz.txt"))) #for older file formats

param_sky = {'fmin': fmin, 'fmax': fmax}

param_obs = {'file_list': beam_list,\
             'gamma_file': rt_file,\
             'phi_array': phi_array,\
             'theta_array': theta_array,\
             'fmin':fmin,\
             'fmax':fmax,\
             'SITE_LATITUDE':SITE_LATITUDE,\
             'SITE_LONGITUDE':SITE_LONGITUDE,\
             'ELEVATION':ELEVATION,\
             'obstimes':obstimes,\
             'include_gamma':True,\
             'output':True,\
             'file_name': output_name,\
             'path_output': beam_path}

TA = Observations(**param_obs)
TA.make_observations()
end = time.time()
print(f"The time of execution of above program is :{(end-start):.1f} sec")
