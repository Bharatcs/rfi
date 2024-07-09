### GMOSS_HealpixSpectra_quad.ipynb
##
## Generate GMOSS spectra towards 3072 Healpix directions
## Using parameters file in GMOSS_params_allpix.txt
##
## Based on code in GMOSS_8aug17.c
##
## Uses a freq_array defined in section below as the frequency axis

import numpy as np
import matplotlib.pyplot as plt
import math
from math import exp, sqrt
import csv
import healpy as hp
import aipy as a
import scipy.constants
import scipy.integrate
import scipy.io
import pickle
import datetime
from multiprocessing import Pool
import time

PI = np.pi

FILE_PARAMS = "GMOSS_params_allpix.txt"

import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = "18"


# In[3]:


### Define frequency array
###
nfreq = 71
freq_array = np.linspace(40.0, 110.0, num=nfreq, endpoint=True, dtype=float)  # MHz

print("Number of frequencies: ", len(freq_array))

freq_GHz = freq_array / 1000.0
print(
    " Frequency array ends: ",
    freq_GHz[0],
    freq_GHz[1],
    ".....",
    freq_GHz[-2],
    freq_GHz[-1],
    " GHz. ",
)
freq_GHz_indices = np.linspace(0, nfreq, num=nfreq, endpoint=False, dtype=int)


### Define Nested Healpix sky with nside 16 (3072 pixels)
###
NSIDE = 16  # hardcoded to conform to the content of FILE_PARAMS

npix = hp.pixelfunc.nside2npix(NSIDE)
resol = hp.pixelfunc.nside2resol(NSIDE, arcmin=True)
print(
    "Generate spectra at healpix with nside, npix, resol (arcmin): ", NSIDE, npix, resol
)


# In[5]:


### Read in the parameter file
###
flag = []
norm_param = []
alpha1_param = []
dalpha_param = []
nubrk_param = []
Tx_param = []
Te_param = []
nut_param = []

with open(FILE_PARAMS) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row != []:
            row = row[0].split()

            flag.append(int(row[1]))
            norm_param.append(float(row[2]))
            alpha1_param.append(float(row[3]))
            dalpha_param.append(float(row[4]))
            nubrk_param.append(float(row[5]))
            Tx_param.append(float(row[6]))
            Te_param.append(float(row[7]))
            nut_param.append(float(row[8]))

print("Got parameters in ", len(Tx_param), " pixels.")


# In[6]:


### Definitions

band_definition = [
    [40, 110.0]
]

GSPAN0 = 100.0  # defines a factor by which the range of integration is expanded beyond required boundary
# this is an initial guess and is updated for each sky pixel to avoid discontinuities in method

q_e = 1.6e-19
Bmag = 1e-9  # Tesla == 10 micro-Gauss
sin_alph = 1.0
m_e = 9.1e-31
cvel = scipy.constants.c
scale_gam_nu = (3.0 * q_e * Bmag * sin_alph) / (4.0 * PI * m_e * cvel)

# Return value of polynomial
func_poly = lambda p, x: (np.polyval(p, x))


def modbessik2(u):  # Modified Bessel Fn of second kind
    # for non-negative real fractional order for real positive argument

    uu = max(0.01, u)  # np.where ((u<0.01),0.01,u)
    bk = scipy.special.kve(5.0 / 3.0, 1 / uu, out=None)
    return bk / (exp(1 / uu) * uu * uu)


def fofx1float(gama, nu, scale_gam_nu, C1, fmult):

    nu_c = (gama * gama) * (scale_gam_nu / 1e9)
    x = nu / nu_c
    rint = scipy.integrate.quad(modbessik2, 0.0, (1.0 / x))
    p1 = -((2 * C1) - 3.0)
    return fmult * rint[0] * (gama ** p1) * (x)


def integral_func1(p1, p2, a1, a2, a3, a4):
    return (
        scipy.integrate.quad(
            fofx1float, p1, p2, epsabs=1.0e-4 / a4, epsrel=1e-08, args=(a1, a2, a3, a4)
        )
    )[0]


integ1 = np.vectorize(integral_func1, excluded=["p1", "p2", "a2", "a3", "a4"])
integ2 = np.vectorize(integral_func1, excluded=["p1", "p2", "a2", "a3", "a4"])

import warnings

warnings.filterwarnings("ignore")


# In[7]:


## Iterate over sky pixels, computing the entire spectrum at each sky position


def compute_pixel_spectra(j):
    Tx = 10.0 ** Tx_param[j]
    Te = 10.0 ** Te_param[j]
    nu_t = 10.0 ** nut_param[j]
    alpha1 = 10.0 ** alpha1_param[j]

    extn = np.exp(-1.0 * ((nu_t / freq_GHz) ** 2.1))

    if flag[j] == 0:  ## the case where alpha2 steeper than alpha1

        fnorm = 10.0 ** norm_param[j]
        alpha2 = alpha1 + 10.0 ** dalpha_param[j]
        nu_break = 10.0 ** nubrk_param[j]
        gama_break = np.sqrt((nu_break) / scale_gam_nu)
        xb = gama_break

        nu_min0 = freq_GHz[0] * 1e9 / GSPAN0
        nu_max0 = freq_GHz[-1] * 1e9 * GSPAN0
        xl0 = np.sqrt((nu_min0) / scale_gam_nu)
        xu0 = np.sqrt((nu_max0) / scale_gam_nu)

        if xl0 > xb or xu0 < xb:

            pflag = 0

            if xl0 > xb:
                C1 = alpha2
            else:
                C1 = alpha1
            fmult = (gama_break) ** (2 * C1 - 3)

            xl = np.sqrt((freq_GHz * 1e9) / (GSPAN0 * scale_gam_nu))
            xu = np.sqrt((freq_GHz * 1e9 * GSPAN0) / scale_gam_nu)

            index_array = np.transpose([xl, xu, freq_GHz])

            cspectT = [
                (
                    scipy.integrate.quad(
                        fofx1float,
                        xl[i],
                        xu[i],
                        args=(freq_GHz[i], scale_gam_nu, C1, fmult),
                    )
                )[0]
                for i in freq_GHz_indices
            ]
            cspect = fnorm * (
                (freq_GHz ** -2.0) * cspectT + Tx * (freq_GHz ** -2.1)
            ) * extn + Te * (1.0 - extn)

        else:

            pflag = 1

            xl1 = xl0
            xu1 = xb
            fmult1 = (gama_break) ** (2 * alpha1 - 3)
            xl2 = xb
            xu2 = xu0
            fmult2 = (gama_break) ** (2 * alpha2 - 3)

            cspectT1 = integ1(xl1, xu1, freq_GHz, scale_gam_nu, alpha1, fmult1)
            cspectT2 = integ1(xl2, xu2, freq_GHz, scale_gam_nu, alpha2, fmult2)

            cspect = fnorm * (
                (freq_GHz ** -2.0) * np.add(cspectT1, cspectT2)
                + Tx * (freq_GHz ** -2.1)
            ) * extn + Te * (1.0 - extn)

    else:  ##   If data requires alpha2 flatter than alpha1,
        ##   then model as sum of power laws, i.e. steep and flat spectrum sources

        pflag = 2

        fnorm1 = norm_param[j]
        fnorm2 = nubrk_param[j]
        alpha2 = alpha1 - 10.0 ** dalpha_param[j]

        cspect = fnorm1 * (
            (freq_GHz ** (-alpha1))
            + fnorm2 * (freq_GHz ** (-alpha2))
            + Tx * (freq_GHz ** -2.1)
        ) * extn + Te * (1.0 - extn)

    print(
        "At pixel: ",
        j,
        " of ",
        npix,
        " flag: ",
        pflag,
        " at time: ",
        datetime.datetime.now(),
    )
    return cspect


# In[8]:


pool = Pool(processes=5)
i = range(0, npix)
results = pool.map(compute_pixel_spectra, i)
pool.close()
pool.join()


# In[16]:


spectrum = np.array(results)
print(spectrum.shape)

np.savetxt("GMOSS_saras_freq.txt", spectrum)