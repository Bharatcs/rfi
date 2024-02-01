import numpy as np
import pandas as pd
import glob

def DBcleanup(csv_files):
	""" Corrects co-ordinates for data obtained from FM list
	Args: glob of all csv files 

	Returns: Corrects the coordinates and rewrites the corrections into the original file
	"""	
	for csv_file in csv_files:
		temp_df=pd.read_csv(csv_file)
    		temp_df=temp_df.dropna()
    		lat=temp_df['Latitude in degrees'].to_numpy()
    		lon=temp_df['Longitude in degrees'].to_numpy()
    		corrlat=[]
    		corrlon=[]
    		for i in lon:
        		sign = -1 if i < 0 else 1  # Determine the sign of the angle
        		degrees = int(i)
       			arcminutes = int(abs((i - degrees) * 100))
        		arcseconds = (abs(i) - abs(degrees) - (abs(arcminutes )/100))*10000 
        		corrlon.append(sign * (abs(degrees) + arcminutes / 60 + arcseconds / 3600))
    		for j in lat:
        		degrees1 = int(j)
        		sign1 = -1 if degrees1 < 0 else 1  # Determine the sign of the angle
        		arcminutes1 = int(abs((j - degrees1) * 100))
        		arcseconds1 = (abs(j) - abs(degrees1) - (abs(arcminutes1 )/100))*10000 
        		corrlat.append(sign1 * (abs(degrees1) + arcminutes1 / 60 + arcseconds1 / 3600))
    	temp_df['Latitude in degrees']=corrlat
    	temp_df['Longitude in degrees']=corrlon
    	temp_df.to_csv(csv_file)



