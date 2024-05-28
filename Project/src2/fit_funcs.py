import numpy as np
# from astropy.coordinates import SkyCoord
# import astropy.units as u
# from astropy.coordinates import FK5
# from astropy.time import Time, TimeDelta
# from astropy.coordinates import SkyCoord, EarthLocation, AltAz
# import healpy as hp
# import matplotlib.pyplot as plt
# from astropy.coordinates import Angle
# from astropy import units
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate
import copy
import os
import sys
from tqdm.notebook import tqdm
#from tqdm import tqdm
import glob
import h5py
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["font.size"] = "18"



#path_ms1 = '/home/pratush/Documents/saras3/'
#path_ms2 = '/home/pratush/Documents/saras3/utils'
#sys.path.append(path_ms1)
#sys.path.append(path_ms2)

#from ms_fit_joint import *
from scipy.optimize import basinhopping
import scipy.io

# %matplotlib notebook
# #%matplotlib inline
# -

cwd = os.getcwd()


##rescaling
def rescale(arr, min1=-1, max1=1, log=True): ##scales x axis to -1 to +1
    arr = np.asfarray(arr)
    if log==True:
        arr = np.log10(arr)
    min_arr = np.amin(arr)
    max_arr = np.amax(arr)
    arr_sc  = ((max1 - min1))*((arr) - float(min_arr))/(max_arr - min_arr) + min1
    return arr_sc


def check_smoothness(p, x, keju_inf):

    nterms = len(p)
    c1 = np.poly1d(p)
    MM = 2

    if nterms <= 3:
        return True

    for i in range(MM, (nterms-1)):  # index 2 here implies quadratic is allowed

        c2 = np.polyder(c1,i)
        der = c2(x)

        no_zero_crossing = ((der[:-1] * der[1:]) < 0).sum() - keju_inf
        if no_zero_crossing > 0:
            return False
    return True



# def check_smoothness_one_inf(p, x):
#
#     nterms = len(p)
#     c1 = np.poly1d(p)
#     MM = 2
#
#     if nterms <= 3:
#         return True
#
#     for i in range(MM, (nterms-1)):  # index 2 here implies quadratic is allowed
#
#         c2 = np.polyder(c1,i)
#         der = c2(x)
#
#         no_zero_crossing = ((der[:-1] * der[1:]) < 0).sum() - 1
#         if no_zero_crossing > 0:
#             return False
#     return True

# +
##Polymodel

def model_log_log(p, x):
    
    x_res = rescale(x)
    

    model = 10**(np.polyval(p,x_res))
    return model 

def chisq_poly(p, *args):
    #print('came here')
    x, y, e, gloss,keju_inf= args
    
    wtt     = 1/e**2
    if gloss == True:
        if check_smoothness(p,rescale(x),keju_inf) == False:
            return 1e14
            
        
    
    model_data = model_log_log(p, x)
    cc = np.sum(wtt*(y-model_data)**2)/np.sum(wtt)
    return cc

def chi_dunk(x1, y1, p00, e1, gloss =True, keju_inf = 0,iterations=100):
    print('initial_6')
    
    stepsize         = 1e-5
    T                = 1e-5
    niter            = 1
    seed             = 1

    args             = (x1, y1, e1,gloss,keju_inf)

    OPTIONS={'fatol':1e-10, 'xatol':1e-10, 'maxiter':1e5, 'maxfev':1e5}
    #OPTIONS={ 'maxiter':1e5, 'maxfev':1e5}
    minimizer_kwargs={'method':'Nelder-Mead', 'args':args, 'options':OPTIONS}

    for i in tqdm(range(iterations)):
        p_out = basinhopping(chisq_poly, p00, minimizer_kwargs=minimizer_kwargs,T=T, \
                             stepsize=stepsize, niter=niter, seed=seed)
        p00 = p_out.x

    p_out = basinhopping(chisq_poly, p00, minimizer_kwargs=minimizer_kwargs,T=T, \
                         stepsize=stepsize, niter=niter, seed=seed)
    p1 = p_out.x

    print ("Coefficients : {} \n".format(p1))
    print("Residual rms : {}".format(np.sqrt(chisq_poly(p1, *args))))

    p1_basin    = copy.deepcopy(p1)
    res_chi = np.sqrt(chisq_poly(p1_basin, *args))
    y_fit_basin = model_log_log(p1_basin, x1)

    yres_basin  = y1 - y_fit_basin

    return p1_basin, yres_basin, y_fit_basin,res_chi
# -



# +
##Poly plus gaussain Model

def make_gauss(A,m,sg,x):
    gauss = A*(np.exp((-(x-m)**2)/(2*(sg**2))))
    return gauss
    

def model_log_log_G(p, x, chh=None ,chl=0):
    if chh==None:
        chh = len(x)
    x_res = rescale(x)
    x_res = x_res[chl:chh]
    

    P_log = np.polyval(p[:-3],x_res)
    Gaussian =  p[-3]*(np.exp(-((x[chl:chh]-p[-2])**2)/(2*(p[-1]**2))))
    model = 10**(P_log) + Gaussian
    return model
    #model = 10**(np.polyval(p[:-3], rescale(x))) + p[-3]*(np.exp(-((x-p[-2])**2)/(2*(p[-1]**2))))
    #return model

def chisq_poly_G(p, *args):
    x, y, gloss, keju_inf, e , chh, chl = args
    if chh==None:
        chh = len(x)
    if p[-2] > max(x) or p[-2]<min(x) :
        return 1e14
    
    if p[-1] > max(x)-min(x) or p[-1] < 1:
        return 1e14
    #if check_smoothness(p[:-3],rescale(x)) == False:
        #return 1e14
    if gloss == True:
        if check_smoothness(p[:-3],rescale(x),keju_inf) == False:
            return 1e14
    
    wtt     = 1/e**2
    model_data = model_log_log_G(p, x, chh, chl)

    cc = np.sum(wtt*(y-model_data)**2)/np.sum(wtt)
    #print(cc)
    return cc

def chi_dunk_G(x1, y1, p00, gloss = True, keju_inf = 0, e1=None, chh=None, chl=0 ,iterations=50):
    if chh==None:
        chh = len(x1)
    #y1= y1[chl:chh]
    if e1==None:
        e1 = np.ones(chh-chl)/100
        

    stepsize         = 1e-5
    T                = 1e-5
    niter            = 10
    seed             = 1

    args             = (x1, y1, gloss, keju_inf,e1, chh, chl)

    #OPTIONS={'ftol':1e-10, 'xtol':1e-10, 'maxiter':1e5, 'maxfev':1e5}
    OPTIONS={'maxiter':1e5, 'maxfev':1e5}
    minimizer_kwargs={'method':'Nelder-Mead', 'args':args, 'options':OPTIONS}
    #p1 = np.zeros([iterations, len(p00)])
    #y_fit_basin = np.zeros([iterations, len(x1)])
    #yres_basin = np.zeros([iterations, len(x1)])
    for i in tqdm(range(iterations)):
        pout = basinhopping(chisq_poly_G, p00, minimizer_kwargs=minimizer_kwargs,T=T, \
                            stepsize=stepsize, niter=niter, seed=seed)
        p00 = pout.x
        
    pout = basinhopping(chisq_poly_G, p00, minimizer_kwargs=minimizer_kwargs,T=T, \
                        stepsize=stepsize, niter=niter, seed=seed)

        #p1[i] = pout.x
        #p1_basin    = copy.deepcopy(p1[i])
        #y_fit_basin[i] = model_log_log_G(p1_basin, x1)
        #yres_basin[i]  = y1 - y_fit_basin[i]
    p1 = pout.x
    
    #pout = basinhopping(chisq_poly_sine, p00, minimizer_kwargs=minimizer_kwargs,T=T,\
    #stepsize=stepsize, niter=niter, seed=seed)
    

    print ("Coefficients : {} \n".format(p1))
    print("Residual rms : {}".format(np.sqrt(chisq_poly_G(p1, *args))))

    p1_basin    = copy.deepcopy(p1)
    res_chi = np.sqrt(chisq_poly_G(p1_basin, *args))
    y_fit_basin = model_log_log_G(p1_basin, x1)

    yres_basin  = y1 - y_fit_basin[chl:chh]
    
    print('Poly Paramaters:',p1_basin[:-3])
    
    print('Gaussian Paramaters:',p1_basin[-3:])
    return p1_basin , yres_basin, y_fit_basin ,res_chi


# -

def crap_check(a,b):
    return a+b+a+b


def all_var_res_plotter(chi_res_dict):
    ft_type_list = list(chi_res_dict.keys())
    plt.figure(figsize=(7,7))
    matplotlib.rcParams.update({'font.size': 10})

    plt.title('Comparison fit residuals')
    while True:
        ad_plt = input('Do you want to add plot?Y/N')
        if ad_plt == 'N':
            print('\033[1m'+'finally! Here is your plot'+'\033[0m')
            break
        if ad_plt =='Y':
            print('\033[1m'+'Ah shit! here we go again! Alright pick one.'+'\033[0m')
            
        pprint.pprint(list(enumerate(ft_type_list)))
        ft_typ = int(input('what type of fit?'))
        vari_list = list(chi_res[list(chi_res.keys())[ft_typ]]['residual'].keys())
        pprint.pprint(list(enumerate(vari_list)))
        vari = int(input('Which variation?'))
        plt.plot(freq_full[0:51], \
                 chi_res_dict[ft_type_list[ft_typ]]['residual'][vari_list[vari]], \
                 label= ft_type_list[ft_typ]+' ' +vari_list[vari])


    plt.legend()
    plt.show()


# +
##Poly plus scale factor Model

def model_log_log_scale(p, x, G_S_arr):
    
    x_res = rescale(x)
    

    P_log = np.polyval(p[:-1],x_res)
    model = 10**(P_log) + p[-1]*G_S_arr
    return model

def chisq_poly_scale(p, *args):
    x, y, G_S_arr,gloss,keju_inf, e = args
    #if p[-2] > max(x) or p[-2]<min(x) :
        #return 1e14
    
    #if p[-1] > max(x)-min(x) or p[-1] < 1:
        #return 1e14
    wtt     = 1/e**2
    
    if gloss == True:
        if check_smoothness(p[:-1],rescale(x),keju_inf) == False:
            return 1e14
    
    model_data = model_log_log_scale(p, x,G_S_arr)

    cc = np.sum(wtt*(y-model_data)**2)/np.sum(wtt)
    #print(cc)
    return cc

def chi_dunk_scale(x1, y1, G_S_arr, p00, e1, gloss = True,keju_inf = 0, iterations=100):
    print('initial_scale')
    
        

    stepsize         = 1e-5
    T                = 1e-5
    niter            = 1
    seed             = 1

    args             = (x1, y1, G_S_arr, gloss , keju_inf, e1)

    OPTIONS={'fatol':1e-10, 'xatol':1e-10, 'maxiter':1e5, 'maxfev':1e5}
    #OPTIONS={'maxiter':1e5, 'maxfev':1e5}
    minimizer_kwargs={'method':'Nelder-Mead', 'args':args, 'options':OPTIONS}
    #p1 = np.zeros([iterations, len(p00)])
    #y_fit_basin = np.zeros([iterations, len(x1)])
    #yres_basin = np.zeros([iterations, len(x1)])
    for i in tqdm(range(iterations)):
        pout = basinhopping(chisq_poly_scale, p00, minimizer_kwargs=minimizer_kwargs,T=T, \
                            stepsize=stepsize, niter=niter, seed=seed)
        p00 = pout.x
        
    pout = basinhopping(chisq_poly_scale, p00, minimizer_kwargs=minimizer_kwargs,T=T, \
                        stepsize=stepsize, niter=niter, seed=seed)

    p1 = pout.x
    

    print ("Coefficients : {} \n".format(p1))
    print("Residual rms : {}".format(np.sqrt(chisq_poly_scale(p1, *args))))

    p1_basin    = copy.deepcopy(p1)
    res_chi = np.sqrt(chisq_poly_scale(p1_basin, *args))
    y_fit_basin = model_log_log_scale(p1_basin, x1,G_S_arr)

    yres_basin  = y1 - y_fit_basin
    
    print('Poly Paramaters:',p1_basin[:-1])
    
    print('Scale Factor:',p1_basin[-1])
    return p1_basin , yres_basin, y_fit_basin, res_chi



## introducing DC component

#'''''
#this following piece of code has a DC component in the model

#'''''

#''''
#this is the scale factor model with a DC component
#''''


def model_log_log_scale_DC(p, x, G_S_arr):
    
    x_res = rescale(x)
    

    P_log = np.polyval(p[:-2],x_res)
    model = 10**(P_log) + p[-2] + p[-1]*G_S_arr
    return model

def chisq_poly_scale_DC(p, *args):
    x, y, G_S_arr,gloss,keju_inf, e = args
    #if p[-2] > max(x) or p[-2]<min(x) :
        #return 1e14
    
    #if p[-1] > max(x)-min(x) or p[-1] < 1:
        #return 1e14
    wtt     = 1/e**2
    
    if gloss == True:
        if check_smoothness(p[:-2],rescale(x),keju_inf) == False:
            return 1e14
    
    model_data = model_log_log_scale_DC(p, x,G_S_arr)

    cc = np.sum(wtt*(y-model_data)**2)/np.sum(wtt)
    #print(cc)
    return cc

def chi_dunk_scale_DC(x1, y1, G_S_arr, p00, e1, gloss = True,keju_inf = 0, iterations=100):
    print('initial_scale_DC')
    
        

    stepsize         = 1e-5
    T                = 1e-5
    niter            = 1
    seed             = 1

    args             = (x1, y1, G_S_arr, gloss , keju_inf, e1)

    OPTIONS={'fatol':1e-10, 'xatol':1e-10, 'maxiter':1e5, 'maxfev':1e5}
    #OPTIONS={'maxiter':1e5, 'maxfev':1e5}
    minimizer_kwargs={'method':'Nelder-Mead', 'args':args, 'options':OPTIONS}
    #p1 = np.zeros([iterations, len(p00)])
    #y_fit_basin = np.zeros([iterations, len(x1)])
    #yres_basin = np.zeros([iterations, len(x1)])
    for i in tqdm(range(iterations)):
        pout = basinhopping(chisq_poly_scale_DC, p00, minimizer_kwargs=minimizer_kwargs,T=T, \
                            stepsize=stepsize, niter=niter, seed=seed)
        p00 = pout.x
        
    pout = basinhopping(chisq_poly_scale_DC, p00, minimizer_kwargs=minimizer_kwargs,T=T, \
                        stepsize=stepsize, niter=niter, seed=seed)

    p1 = pout.x
    

    print ("Coefficients : {} \n".format(p1))
    print("Residual rms : {}".format(np.sqrt(chisq_poly_scale_DC(p1, *args))))

    p1_basin    = copy.deepcopy(p1)
    res_chi = np.sqrt(chisq_poly_scale_DC(p1_basin, *args))
    y_fit_basin = model_log_log_scale_DC(p1_basin, x1,G_S_arr)

    yres_basin  = y1 - y_fit_basin
    
    print('Poly Paramaters:',p1_basin[:-2])
    print('DC_component:',p1_basin[-2])
    print('Scale Factor:',p1_basin[-1])
    return p1_basin , yres_basin, y_fit_basin, res_chi



#''''
#This is log_log model with DC component
#''''


def model_log_log_DC(p, x):
    
    x_res = rescale(x)
    

    model = 10**(np.polyval(p[:-1],x_res)) + p[-1]
    return model 

def chisq_poly_DC(p, *args):
    #print('came here')
    x, y, e, gloss,keju_inf= args
    
    wtt     = 1/e**2
    if gloss == True:
        if check_smoothness(p[:-1],rescale(x),keju_inf) == False:
            return 1e14
            
        
    
    model_data = model_log_log_DC(p, x)
    cc = np.sum(wtt*(y-model_data)**2)/np.sum(wtt)
    return cc

def chi_dunk_DC(x1, y1, p00, e1, gloss =True, keju_inf = 0,iterations=100):
    print('initial_DC')
    
    stepsize         = 1e-5
    T                = 1e-5
    niter            = 1
    seed             = 1

    args             = (x1, y1, e1,gloss,keju_inf)

    OPTIONS={'fatol':1e-10, 'xatol':1e-10, 'maxiter':1e5, 'maxfev':1e5}
    #OPTIONS={ 'maxiter':1e5, 'maxfev':1e5}
    minimizer_kwargs={'method':'Nelder-Mead', 'args':args, 'options':OPTIONS}

    for i in tqdm(range(iterations)):
        p_out = basinhopping(chisq_poly_DC, p00, minimizer_kwargs=minimizer_kwargs,T=T, \
                             stepsize=stepsize, niter=niter, seed=seed)
        p00 = p_out.x

    p_out = basinhopping(chisq_poly_DC, p00, minimizer_kwargs=minimizer_kwargs,T=T, \
                         stepsize=stepsize, niter=niter, seed=seed)
    p1 = p_out.x

    print ("Coefficients : {} \n".format(p1))
    print("Residual rms : {}".format(np.sqrt(chisq_poly_DC(p1, *args))))

    p1_basin    = copy.deepcopy(p1)
    res_chi = np.sqrt(chisq_poly_DC(p1_basin, *args))
    y_fit_basin = model_log_log_DC(p1_basin, x1)
    
    print ("DC_component:",p1_basin[-1] )

    yres_basin  = y1 - y_fit_basin

    return p1_basin, yres_basin, y_fit_basin,res_chi


#getting 3D_beams from the files

