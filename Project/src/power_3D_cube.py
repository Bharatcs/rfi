import numpy as np
import pandas as pd
from src.satellite_properties import get_beam_pattern
pd.options.mode.chained_assignment = None

def get_dataframe_fov(Rx_Power_in_Kelvin, df, found_common, Rad_of_FOV, elev_angle):
    """Create dataframe for the FOV of the satellite for each pixel position at different altitudes

    Args:
        Rx_Power_in_Kelvin (array): An array of values of received RFI (FM) power in Kelvin
        df (dataframe): dataframe of the database (FM)
        found_common (array): An array of indices of the pixel number that intersect between the set of pixel numbers in the 
    FOV circle/disc and location of FM transmitters in the database
        Rad_of_FOV (array): An array of values of radius of FOV of the satellite in radians
        elev_angle (array): An array of the elevation angle of the satellite's antenna beam

    Returns:
        dataframe: Dataframes for the FOV of the satellite for each pixel position at different altitudes
    """
    df_data=pd.DataFrame(Rx_Power_in_Kelvin)
    df_data['Pixel_number']=df['Pixel_number'].values
    df_data['New Frequency']=df['New Frequency'].values
    
    df_fov=np.zeros((len(found_common),len(Rad_of_FOV)),dtype=object)

    for j in range(len(Rad_of_FOV)):
        for i in range(len(found_common)):
            df_fov[i][j]= df_data.loc[df_data['Pixel_number'].isin(found_common[i][j])]
            df_fov[i][j][j]=df_fov[i][j][j]*get_beam_pattern("cos square", elev_angle[j,i,found_common[i,j]])
            df_fov[i][j]= df_fov[i][j].groupby(['New Frequency']).sum()  # df9['Pixel_number']= Column consisting of 
            df_fov[i][j]= df_fov[i][j].reset_index()
    return df_fov

def get_power_output(npix, Rad_of_FOV, freq_range, df_fov):
    """Create a three dimensional data cube of RFI (FM) power having dimensions N_altitude x N_pixel x N_frequency

    Args:
        npix (array):  Array of evenly spaced pixel numbers
        Rad_of_FOV (array): An array of values of radius of FOV of the satellite in radians
        freq_range (array): User-defined frequency range
        df_fov (dataframe): Dataframes for the FOV of the satellite for each pixel position at different altitudes

    Returns:
        array: A three dimensional array of values of RFI (FM) power
    """
    # Initializing the 3D array to store the received power for each pixel,
    # each frequency and at different altitudes ###
    power_output=np.zeros((len(Rad_of_FOV),npix,len(freq_range)),dtype=object)

    # Create the 3D array to store the received power for each pixel, each frequency and at different altitudes 
    # k: loops along the length of the number of altitude
    # m: loops along the length of the number of pixels based on the given NSIDE
    # l: loops along the length of the number of frequencies in the frequency axis defined by user
    for k in range(len(Rad_of_FOV)):
        for m in range(npix):    
            for l in range(len(freq_range)):
                # checking for empty dataframes
                if (df_fov[m][k][k][df_fov[m][k]['New Frequency'] == freq_range[l]].values).size==0:
                    power_output[k][m][l]=0
                else: 
                    power_output[k][m][l]=float(df_fov[m][k][k][df_fov[m][k]['New Frequency'] == freq_range[l]].values)
    return power_output
