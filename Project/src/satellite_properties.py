import numpy as np
from astropy.constants import R_earth

def get_altitudes():
    """Returns the user-defined altitudes of the satellite in logspace.

    Returns:
        array: An array of altitudes in km in logspace.
    """
    # The default value for minimum and maximum altitudes are 400km (Low-earth orbit)
    # and 36000km (Geo-stationary orbit) respectively. The default value for data_points is 3.
    alt_min = float(input("Enter the minimum altitude of the satellite: "))
    alt_max = float(input("Enter the maximum altitude of the satellite: "))
    data_points = int(input("Enter number of data points : "))
    altitudes = np.logspace(np.log10(alt_min), np.log10(alt_max), data_points) 
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
    x_ang=np.zeros((npix, npix))
    y_ang=np.zeros((npix, npix))
    elev_ang=np.zeros((len(altitudes),npix,npix))
    R_E = R_earth.to('km').value
    for k in range(len(altitudes)):
        for i in range(len(theta)):
            for j in range(len(theta)):
                x_ang[i,j]=((np.cos(np.radians(theta[i])))*(np.cos(np.radians(theta[j])))*
                            (np.cos(np.radians(phi[j]-phi[i])))+(np.sin(np.radians(theta[i])))*
                            (np.sin(np.radians(theta[j]))))
                y_ang[i,j]=(np.arccos(x_ang[i,j]))
                B=(altitudes[k]+R_E)/R_E
                elev_ang[k,i,j]=-(np.degrees(np.arctan((B-np.cos(np.radians(y_ang[i,j])))/np.sin(np.radians(y_ang[i,j])))))
    return elev_ang

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
