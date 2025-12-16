import numpy as np
from astropy.constants import R_earth

def get_altitudes():
    """Returns the user-defined altitudes of the satellite in logspace.

    Returns:
        array: An array of altitudes in km in logspace.
    """
    # The default value for minimum and maximum altitudes are 400km (Low-earth orbit)
    # and 36000km (Geo-stationary orbit) respectively. The default value for data_points is 3.
    d1,d2,d3=(input('Enter desired altitudes:')).split()
    altitudes =np.array((d1,d2,d3)).astype('float64')
    return altitudes

def get_altitudes_manual(d1,d2,d3):
    """Returns the user-defined altitudes of the satellite in logspace.

    Returns:
        array: An array of altitudes in km in logspace.
    """
    altitudes =np.array((d1,d2,d3)).astype('float64')
    return altitudes

def calc_elev_angle(npix, phi, theta, altitudes):
    """Calculation of the elevation angle (theta). The elevation angle is the angle between the 
    satellite's azimuthal plane and the FM transmitting station on the Earth.

    Args:
        npix (int): Number of pixels of healpy map
        phi (array): Array of the angular coordinate longitude
        theta (array): Array of the angular coordinate co-latitude
        altitudes (array): user-defined altitudes of the satellite

    Returns:
        array: An array of the elevation angle of the satellite's antenna beam
    """
    eps = 1e-12
    elev_ang = np.zeros((len(altitudes), npix, npix))
    R_E = R_earth.to('km').value
    for k in range(len(altitudes)):
        for i in range(len(theta)):
            for j in range(len(theta)):
                cos_theta_i = np.cos(np.radians(theta[i]))
                cos_theta_j = np.cos(np.radians(theta[j]))
                sin_theta_i = np.sin(np.radians(theta[i]))
                sin_theta_j = np.sin(np.radians(theta[j]))
                cos_phi_diff = np.cos(np.radians(phi[j] - phi[i]))
                
                x_ang = cos_theta_i * cos_theta_j * cos_phi_diff + sin_theta_i * sin_theta_j
                x_ang = np.clip(x_ang, -1.0, 1.0)
                y_ang = np.arccos(x_ang)
                
                B = (altitudes[k] + R_E) / R_E
                sin_y_ang = np.sin(y_ang)
                
                elev_ang[k, i, j] = -(np.degrees(np.arctan((B - cos_theta_j) / (sin_y_ang+eps))))
    return elev_ang

import numpy as np

def calc_elev_angle_fast(npix, phi, theta, altitudes, eps=1e-12):
    """
    Fully vectorized calculation of elevation angle.
    Returns array of shape (len(altitudes), npix, npix).
    """
    # convert to radians once
    phi_rad = np.radians(phi)
    theta_rad = np.radians(theta)

    # precompute trig for theta (length n)
    cos_theta = np.cos(theta_rad)   # shape (n,)
    sin_theta = np.sin(theta_rad)   # shape (n,)

    # cos(phi_j - phi_i) as an (n,n) matrix where element (i,j) is phi[j]-phi[i]
    cos_phi_diff = np.cos(phi_rad[None, :] - phi_rad[:, None])  # shape (n,n)

    # x_ang (n,n)
    x_ang = (cos_theta[:, None] * cos_theta[None, :] * cos_phi_diff +
             sin_theta[:, None] * sin_theta[None, :])
    x_ang = np.clip(x_ang, -1.0, 1.0)

    # angular separation y_ang (n,n)
    y_ang = np.arccos(x_ang)
    sin_y_ang = np.sin(y_ang)  # shape (n,n)

    # vectorize over altitudes: B has shape (m,)
    R_E = R_earth.to('km').value
    B = (np.asarray(altitudes) + R_E) / R_E   # shape (m,)

    # cos_theta_j needs shape (1, 1, n) to broadcast with sin_y_ang (1,n,n)
    cos_theta_j = cos_theta[None, None, :]   # shape (1,1,n)
    sin_y_ang_expanded = sin_y_ang[None, :, :]  # shape (1,n,n); matches (m,n,n) when B expanded

    # compute numerator (B - cos_theta_j) with broadcasting and then arctan
    numer = B[:, None, None] - cos_theta_j    # shape (m, n, n) via broadcasting
    elev = -np.degrees(np.arctan(numer / (sin_y_ang_expanded + eps)))  # shape (m,n,n)

    return elev


def get_beam_pattern(beam, theta):
    """Returns the beam pattern of the satellite antenna beam. The default beam pattern is
    assumed to be cos^2 (theta) measuring downward, where theta is the elevation angle w.r.t. the satellite azimuthal plane.
    The beam pattern is frequency independent and the azimuthal angle is assumed to be 0.

    Args:
        beam (str): beam pattern ("cos square" or "sin square") of the satellite (or the receiving antenna)
        theta (array): elevation angle of satellite's antenna beam
        
    Returns:
        array: An array of values of radiation pattern of the satellite antenna beam
    """
    if beam == "cos square":
        pattern = np.cos(np.radians(theta))**2
    elif beam == "sin square":
        pattern = np.sin(np.radians(theta))**2 
    else:
        print("Beam not found")
    return pattern

def calc_field_of_view(altitudes):
    """Calculates the field of view (FOV) of the satellite at user-defined altitudes.

    Args:
        altitudes (array): user-defined altitudes of the satellite in km

    Returns:
        array: An array of values of FOV of the satellite in radians for user-defined altitudes.
    """
    # Considering Nadir-pointing Field of View Geometry
    # Considering the FOV of the satellite to be tangent to the surface of the Earth
    R_E = R_earth.to('km').value
    FOV=np.zeros(len(altitudes))
    for i in range(0,len(altitudes)):
        # Consider a case of full coverage under elevation of 0º
        # Field of view for maximal coverage in radians when elevation is 0º 
        FOV[i]= 2*np.arcsin(R_E/(R_E+ altitudes[i]))  
        print(f"The Field of view of the satellite at a height of {altitudes[i]:.2f} km is {FOV[i]:.2f} radians")
    return FOV

def calc_central_angle(altitudes):
    """Calculates the central angle and radius of the FOV of the satellite at user-defined altitudes.

    Args:
        altitudes (array): user-defined altitudes of the satellite in km

    Returns:
        Central_angle (array): An array of values of central angle of the satellite in radians
        Rad_of_FOV (array): An array of values of radius of FOV of the satellite in radians
    """
    # The surface of the coverage area of the Earth depends on the central angle
    R_E = R_earth.to('km').value
    Central_angle=np.zeros(len(altitudes))
    for i in range(0,len(altitudes)):
        Central_angle[i]=np.arccos(R_E/(R_E+altitudes[i])) # Central angle in radians
        Dia_of_FOV=2*Central_angle*R_E  # Diameter of the FOV (disc on the Earth's surface)in km
        Rad_of_FOV= Dia_of_FOV/2 # Radius of the FOV in km
        Rad_of_FOV=Rad_of_FOV/R_E  # Radius of the FOV in Radians
        print(f"The Radius of the Field of View for a height of {altitudes[i]:.2f} km in radians is {Rad_of_FOV[i]:.2f}")
    return Central_angle, Rad_of_FOV
