import numpy as np
import healpy as hp

def get_disc(nside, phi, theta, Rad_of_FOV, pix_array):
    """Stores indices of the pixel numbers that are inside the circle/disc(FOV) w.r.t. the altitude
    of the satellite

    Args:
        nside (int): NSIDE of the healpy map
        phi (array): Array of the corresponding angular coordinate (longitude)
        theta (array): Array of the corresponding angular coordinate (co-latitude)
        Rad_of_FOV (array): An array of values of radius of FOV of the satellite in radians
        pix_array (array): Array of evenly spaced pixel numbers

    Returns:
        array: An array of indices of the pixel numbers that are inside the circle/disc(FOV) w.r.t. the altitude of the satellite
    """
    # Using ang2vec convert angles that is co-latitude and longitude
    # in radians to 3D position vector
    vec3d = hp.ang2vec(phi,theta,lonlat=True) 
    
    # Array of indices of the pixel number that are inside the 
    # circle/disc specified by vec and radius
    disc=np.zeros((len(pix_array), len(Rad_of_FOV)),dtype=object)
    
    for i in range(len(Rad_of_FOV)):
        for j in range(len(pix_array)):
            disc[j][i]=hp.query_disc(nside, vec3d[j], radius=Rad_of_FOV[i])
    return disc

def get_common_pix(pix_array, Rad_of_FOV, disc, df):
    """Stores indices of the pixel number that intersect between the set of pixel numbers in the 
    FOV circle/disc and the satellite's location or pixel in space overhead

    Args:
        pix_array (array): Array of evenly spaced pixel numbers
        Rad_of_FOV (array): An array of values of radius of FOV of the satellite in radians
        disc (array): An array of indices of the pixel numbers that are inside the circle/disc(FOV) w.r.t. the altitude of the satellite
        df (dataframe): dataframe of the database (FM)

    Returns:
        array: An array of indices of the pixel number in the 
        FOV circle/disc and the satellite's location or pixel in space overhead
    """
    pix_common=np.zeros((len(pix_array),len(Rad_of_FOV)),dtype=object)
    for i in range(len(Rad_of_FOV)):
        for k in range(len(disc[:, i])):
            # Array of indices of the pixel number that are common
            # between the FOV disc and the satellite pixel no
            pix_common[k][i]=np.intersect1d(df['Pixel_number'], disc[k][i])
    return pix_common

def get_common_Tx(pixel_indices, pix_common, Rad_of_FOV):
    """Stores indices of the pixel number that intersect between the set of pixel numbers in the 
    FOV circle/disc and location of FM transmitters in the database

    Args:
        pixel_indices (array): An array of the healpix pixel numbers corresponding to the latitude and longitude in the database (FM)
        pix_common (array): An array of indices of the pixel number in the FOV circle/disc and the satellite's location or pixel in space overhead
        Rad_of_FOV (array): An array of values of radius of FOV of the satellite in radians

    Returns:
        found_common (array): An array of indices of the pixel number that intersect between the set of pixel numbers in the 
    FOV circle/disc and location of FM transmitters in the database
    """
    # Initializing the array to store the pixel number that are common between
    # the FOV disc and the satellite pixel no
    tx_common=np.zeros((len(pix_common),len(Rad_of_FOV)),dtype=object)
    found_common=np.zeros(( len(pix_common),len(Rad_of_FOV)),dtype=object)

    for j in range(len(Rad_of_FOV)):
        for i in range(len(pix_common)):
            tx_common[i][j]=set(pix_common[i][j])
            # Array of indices of the pixel number that are common between the 
            # FOV disc and the satellite pixel no with Tx having same pixel number
            found_common[i][j] = [l for l in pixel_indices if l in tx_common[i][j]]
    return found_common
