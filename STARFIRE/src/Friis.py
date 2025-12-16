import numpy as np
from astropy.constants import k_B, c
import pandas as pd

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


def calc_Friis_fast(df, altitudes, res=244e3):
    """
    Vectorized Friis calculation.

    Args:
        df (DataFrame): must contain columns 'EIRP' and 'Frequency(MHz)'.
        altitudes (array-like): altitudes in km (shape (m,))
        res (float): resolution/bandwidth in Hz (default 244e3)

    Returns:
        Rx_Power (ndarray): shape (n_tx, m_alt) power in Watts
        Rx_Power_in_dBm (ndarray): shape (n_tx, m_alt) power in dBm (or -inf if zero)
        Rx_Power_in_Kelvin (ndarray): shape (n_tx, m_alt) in Kelvin
        Rx_Power_in_dBW (ndarray): shape (n_tx, m_alt) in dBW (or -inf if zero)
    """
    # ensure numpy arrays
    alt = np.asarray(altitudes)  # km, shape (m,)
    if alt.ndim == 0:
        alt = alt.reshape((1,))

    # Convert columns to numeric types once; coerce invalid entries to NaN -> fill with 0
    eirp = pd.to_numeric(df['EIRP'], errors='coerce').fillna(0.).to_numpy(dtype=float)    # Watts (assumed)
    freq_mhz = pd.to_numeric(df['Frequency(MHz)'], errors='coerce').fillna(0.).to_numpy(dtype=float)

    # handle zero frequencies: set wavelength to 0 for freq==0
    freq_hz = freq_mhz * 1e6         # Hz
    with np.errstate(divide='ignore', invalid='ignore'):
        wavelength = np.where(freq_hz > 0, c.value / freq_hz, 0.0)  # meters

    # shapes: eirp (n,), wavelength (n,), alt (m,)
    n = eirp.shape[0]
    m = alt.shape[0]

    # distance R (meters)
    R = alt * 1e3   # km -> m, shape (m,)

    # denominator (4*pi*R)^2 shape (m,)
    denom = (4.0 * np.pi * R) ** 2   # shape (m,)

    # numerator: EIRP * lambda^2 shape (n,)
    numer = eirp * (wavelength ** 2)  # shape (n,)

    # broadcast to (n, m)
    Rx_Power = numer[:, None] / denom[None, :]   # shape (n, m)
    # ensure non-negative (numerical tiny negatives -> 0)
    Rx_Power = np.where(Rx_Power > 0, Rx_Power, 0.0)

    # dBW and dBm: avoid log10 warnings using where
    # produce -inf where Rx_Power == 0 (you can change to np.nan if you prefer)
    positive_mask = Rx_Power > 0
    Rx_Power_in_dBW = np.full_like(Rx_Power, -np.inf, dtype=float)
    Rx_Power_in_dBW[positive_mask] = 10.0 * np.log10(Rx_Power[positive_mask])

    Rx_Power_in_dBm = np.full_like(Rx_Power, -np.inf, dtype=float)
    Rx_Power_in_dBm[positive_mask] = Rx_Power_in_dBW[positive_mask] + 30.0

    # Kelvin: P / (k_B * res) — res in Hz, k_B in J/K, P in W -> K
    denom_kelvin = (k_B.value * res)
    Rx_Power_in_Kelvin = Rx_Power / denom_kelvin

    return Rx_Power, Rx_Power_in_dBm, Rx_Power_in_Kelvin, Rx_Power_in_dBW