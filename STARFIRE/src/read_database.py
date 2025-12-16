import os
import pandas as pd
import numpy as np
import healpy as hp

# df[13195:14104].to_csv("./South_Africa.csv", index=False)
# df[14139:43317].to_csv("./USA.csv", index=False)
# df[0:2088].to_csv("./Germany.csv", index=False)
# df[2088:10530].to_csv("./Canada.csv", index=False)
# df[10530:13195].to_csv("./Australia.csv", index=False)
# df[14104:14139].to_csv("./Tokyo.csv", index=False)

def read_csv(file_path):
    """Reads the CSV file into a pandas dataframe.
    The user may extend or change the input database (FM) as per the requirements and computational resources available.

    Args:
        file_path (str): path to file

    Returns:
        dataframe: A two -dimensional comma-separated values (csv) file is returned with four columns consisting of \n
                   FM trasmitter data for the countries: Canada, Australia, Germany, USA, and South Africa \n
                   Latitude ranges from 90(N.Pole) to 0 (Equator) to -90(S.Pole) \n
                   Longitude ranges between 0 and 360 degree eastwards \n 
                   Frequency of operation in MHz \n
                   EIRP (Effective Isotropic Radiated Power) in Watts
    """
  #  print(file_path)
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(file_path)
  #  print(df)
    # Drop any rows that have missing values in the "Latitude in degrees" column
    df.dropna(subset=["Latitude in degrees"], inplace=True)
    
    return df
import os

def get_countries(n):
    """Returns the list of countries based on user input

    Args:
        n (str or int): If str, "All" returns a list of all countries in the database.
                        If int, specifies the number of countries to input.

    Returns:
        list: A list containing the database (FM) of the user-specified country(ies)
    """
    list_of_countries = []
    if n != "All":
        for i in range(int(n)):
            cName = str(input(f"Enter the name of {i+1} country")) + ".csv"
            if not os.path.exists(f"./database/{cName}"):
                print("Does not exist in the database")
                break
            list_of_countries.append(cName)
    if n == "All":
        all_countries = os.listdir("./database/")
        # Exclude ".ipynb_checkpoints" if it exists in the list
        if ".ipynb_checkpoints" in all_countries:
            all_countries.remove(".ipynb_checkpoints")
        list_of_countries = all_countries
    return list_of_countries


def read_FM_database(countries):
    """Concatenates the dataframe of multiple countries

    Args:
        countries (list): list of countries

    Returns:
        dataframe: A dataframe is returned concatenating the database (FM) of the user-specified country(ies) 
    """
    df = read_csv(f"./database/{countries[0]}")
    for i in range(len(countries)-1):
        df = pd.concat([df, read_csv(f"./database/{countries[i+1]}")], ignore_index=True)
    return df


def get_freq_value(prompt):
    """Returns the user-defined frequency value.
    """
    while True:
        try:
            value = float(input(prompt))
        except ValueError:
            print("There is an error in the input.")
            continue

        if value<=0:
            print("There is an error in the input.The input cannot be zero or negative")
            continue
        else:
            break
    return value

def get_freq_range():
    # The default frequency range is 55 MHz-110 MHz with a frequency resolution of 244 kHz.
    """Returns the frequency range based on min, max and resolution of frequency.

    Returns:
        array: An array of user-defined frequency range in MHz
    """
    fstart = get_freq_value("Please enter the minimum value of the frequency range (in MHz): ")
    fstop = get_freq_value("Please enter the maximum value of the frequency range (in MHz): ")
    fres = get_freq_value("Please enter the resolution of the frequency (in MHz): ")
    freq_range = np.arange(fstart,fstop,fres)
    return freq_range

def get_freq_range_manual(fstart,fstop,fres):

    freq_range = np.arange(fstart,fstop,fres)
    return freq_range

def extract_common_freq(df, freq_range):
    """Returns the dataframe with a new column consisting of the frequencies (from the FM database)\
        based on the user-defined frequency range.

    Args:
        df (dataframe): dataframe of the user-specified country(ies)
        freq_range (array): User-defined frequency range

    Returns:
        dataframe: A dataframe with an additional column specific to user-defined frequency range.
    """
    # Array consisting of frequencies from the dataset
    data_freq=df['Frequency(MHz)'].values

    diff=np.zeros((len(data_freq),len(freq_range)),dtype=object)
    freq_arr=np.zeros(len(data_freq))

    for i in range(len(data_freq)):
        for j in range(len(freq_range)):
            diff[i][j]=abs(data_freq[i]-freq_range[j])
        freq_arr[i]=freq_range[np.argmin(diff[i])]
            
    # Create column to store the new frequencies based on the user defined frequency axis #####
    df['New Frequency']=freq_arr
    return df

def allocate_pixel_num(nside, df):
    """Allocate pixel number to the Latitude and Longitude of each FM transmitter in the database

    Args:
        nside (int): NSIDE of the healpy map
        df (longitude and latitude of dataframe): columns in the dataframe of the FM database

    Returns:
        dataframe: A dataframe specific to user-defined database (FM)
        array: An array of the healpix pixel numbers corresponding to the latitude and longitude in the database (FM)
    """
    pixel_ind = hp.ang2pix(nside, df['Longitude in degrees'].to_numpy() ,df['Latitude in degrees'].to_numpy(), lonlat=True)
    df['Pixel_number'] = pixel_ind
    return df, pixel_ind