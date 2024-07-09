import numpy as np
from astropy.constants import k_B, c


def calc_Friis(df, altitudes):
    """Calculates the received RFI (FM) power in Watts, dBm and Kelvin w.r.t. altitudes of the satellite 
    using the Friis Transmission Equation.

    Args:
        df (dataframe): dataframe of the database (FM)
        altitudes (array): user-defined altitudes of the satellite in km

    Returns:
        Rx_Power (array): An array of values of received RFI (FM) power in Watts
        Rx_Power_in_dBm (array): An array of values of received RFI (FM) power in dBm
        Rx_Power_in_Kelvin (array): An array of values of received RFI (FM) power in Kelvin
        Rx_Power_in_dBW (array): An array of values of received RFI (FM) power in dBW
    """
    # Considering isotropic transmitter and receiver with gain =1 
    res=244*1e3
    Rx_Power=np.zeros((len(df),len(altitudes)))
    Rx_Power_in_Kelvin=np.zeros((len(df),len(altitudes)))
    Rx_Power_in_dBm=np.zeros((len(df),len(altitudes)))
    Rx_Power_in_dBW=np.zeros((len(df),len(altitudes)))
    for i in range(0,len(altitudes)):
        for j in range(0,len(df)):
            wavelength= c.value/(df.iloc[j]['Frequency(MHz)']*1e6)
            # the Friis Transmission Equation
            Rx_Power[j][i]= ((df.iloc[j]['EIRP'])*(wavelength)**2)/(4*np.pi*altitudes[i]*1e3)**2 
            Rx_Power_in_dBm[j][i]= 10.*np.log10( Rx_Power[j][i])+30
            Rx_Power_in_Kelvin[j][i]=Rx_Power[j][i]/(k_B.value*res) # in Kelvin
            Rx_Power_in_dBW[j][i]= 10.*np.log10( Rx_Power[j][i])
    return Rx_Power, Rx_Power_in_dBm, Rx_Power_in_Kelvin, Rx_Power_in_dBW       
