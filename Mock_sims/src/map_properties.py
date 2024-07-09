import numpy as np
import healpy as hp

def get_map(nside):
    """Returns the number of pixels, pixel indices, longitude and co-latitude of healpy map.

    Args:
        nside (int): NSIDE of the healpy map

    Returns:
        int: corresponding number of pixels
        ndarray: Array of evenly spaced pixel numbers
        array: Array of the corresponding angular coordinate longitude (phi)
        array: Array of the corresponding angular coordinate co-latitude (theta)
    """
    npix = hp.nside2npix(nside)

    # Create an an array of pixel numbers with respect to the NSIDE
    pix_array =np.arange(npix) 

    # Array of the angular coordinates co-latitude(theta) and longitude(phi)
    # in degrees with respect to the given NSIDE
    phi, theta = (hp.pix2ang(nside, ipix=pix_array,lonlat=True)) 
    return npix, pix_array, phi, theta
